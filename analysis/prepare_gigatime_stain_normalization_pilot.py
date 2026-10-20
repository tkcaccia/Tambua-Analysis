#!/usr/bin/env python3
"""Prepare a fixed, CPU-only Reinhard-Lab GigaTIME transfer diagnostic.

Only image colour is changed. Pixel coordinates, scanner RGB order, patient
identity and physical spacing remain fixed. This is a development experiment,
not a replacement for the frozen test images or their QC gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


SAMPLES = (
    "TCGA-EV-5903-01Z-00-DX1_2",  # fit, QC fail; same patient as pass
    "TCGA-EV-5903-01Z-00-DX1_4",  # fit, QC pass; preservation control
    "TCGA-G7-A8LD-01Z-00-DX1_7",  # development, QC fail
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB:
            raise ValueError(f"Not tagged scanner RGB: {path}")
        arr = tif.asarray()
    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB source: {path}")
    return arr


def tissue(rgb: np.ndarray) -> np.ndarray:
    out = (rgb.mean(axis=2) < 240) & (np.ptp(rgb, axis=2) > 10)
    if out.mean() < .15:
        raise ValueError("Insufficient tissue for stain normalisation")
    return out


def moments(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[tissue(rgb)]
    return pixels.mean(axis=0), pixels.std(axis=0)


def normalize(rgb: np.ndarray, target_mean: np.ndarray,
              target_std: np.ndarray) -> tuple[np.ndarray, dict]:
    mask = tissue(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source = lab[mask]
    mean, std = source.mean(axis=0), source.std(axis=0)
    if np.any(std < 5) or np.any(target_std < 5):
        raise ValueError("Reference or input lacks usable colour variation")
    mapped = (lab - mean) * (target_std / std) + target_mean
    mapped = cv2.cvtColor(np.clip(mapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 8).clip(0, 1)
    result = np.rint(rgb * (1-alpha[..., None]) + mapped * alpha[..., None]).clip(0, 255).astype(np.uint8)
    return result, {"source_lab_mean": mean.tolist(), "source_lab_std": std.tolist(),
                    "target_lab_mean": target_mean.tolist(),
                    "target_lab_std": target_std.tolist(),
                    "tissue_fraction": float(mask.mean()),
                    "mean_absolute_rgb_change": float(np.abs(result.astype(np.int16)-rgb.astype(np.int16))[mask].mean())}


def run(reference: Path, prepared: Path, inference: Path,
        model_split_map: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    splits = {row["sample_id"]: row["model_split"] for row in
              json.loads(model_split_map.read_text())["records"]}
    if any(splits[sample] not in {"train", "development"} for sample in SAMPLES):
        raise ValueError("Prespecified pilot includes a test patient")
    ref = read_rgb(reference)
    target_mean, target_std = moments(ref)
    output.mkdir(parents=True)
    records = []
    for sample in SAMPLES:
        source = prepared / "samples" / sample / "he.ome.tif"
        mask = inference / sample / "stardist_labels.tif"
        provenance = json.loads((prepared / "samples" / sample / "provenance.json").read_text())
        qc = json.loads((inference / sample / "stardist_qc.json").read_text())
        if not mask.is_file() or provenance["source_color_reference"] != "scanner_svs_rgb":
            raise ValueError(f"Identity or independent StarDist mask unavailable: {sample}")
        rgb = read_rgb(source)
        normalized, stats = normalize(rgb, target_mean, target_std)
        destination = output / sample
        destination.mkdir()
        mpp = float(provenance["mpp"])
        corrected = destination / "normalized.ome.tif"
        tifffile.imwrite(corrected, normalized, ome=True, photometric="rgb",
                         metadata={"axes": "YXS", "PhysicalSizeX": mpp,
                                   "PhysicalSizeY": mpp,
                                   "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"},
                         compression="zlib")
        if not np.array_equal(read_rgb(corrected), normalized):
            raise ValueError(f"Lossless RGB round trip failed: {sample}")
        records.append({"sample_id": sample, "model_split": splits[sample],
                        "original_nuclear_qc": qc["status"],
                        "original_auc": qc["auc"],
                        "original_boundary_contrast": qc["nuclear_boundary_contrast"],
                        "source_mpp_um": mpp, "shape_yx": list(rgb.shape[:2]),
                        "source_rgb_sha256": sha(source),
                        "normalized_rgb_sha256": sha(corrected),
                        "independent_stardist_mask_sha256": sha(mask),
                        **stats})
    report = {"status": "prepared_cpu_only", "reference": str(reference),
              "reference_sha256": sha(reference),
              "method": "Tissue-restricted Reinhard mean/std matching in uint8 CIELAB to one official GigaTIME H&E control, with Gaussian edge blending; no spatial resampling",
              "cases": records,
              "limit": "Fixed training/development-only diagnostic. Original test images and frozen DAPI QC threshold remain untouched."}
    (output / "PREP.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prepared", "inference", "model-split-map", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    result = run(a.reference, a.prepared, a.inference, a.model_split_map, a.output)
    print(json.dumps({"status": result["status"],
                      "samples": [row["sample_id"] for row in result["cases"]]}))
