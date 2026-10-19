#!/usr/bin/env python3
"""Render true registered ORION Hoechst and virtual DAPI image pixels.

Display transforms are for visualization only. The rank correlations in the
companion score use native 16-bit values, not these rendered images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
import tifffile
import zarr


def display(values: np.ndarray, *, virtual: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if virtual:
        probability_like = np.clip(values / 65535.0, 1e-4, 1 - 1e-4)
        values = np.log(probability_like / (1 - probability_like))
    else:
        scale = max(1.0, float(np.quantile(values, .95)))
        values = np.arcsinh(values / scale)
    lo, hi = np.quantile(values, [.01, .997])
    if hi <= lo:
        raise ValueError("Image has no usable display contrast")
    return np.rint(np.clip((values-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)


def render(study: Path, measured_root: Path, scores: Path, output: Path) -> dict:
    summary = json.loads((scores / "summary.json").read_text())
    if summary["patients"] != 10:
        raise ValueError("Pixel score does not cover all ten reserved QC-passing patients")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = []
    contacts: list[tuple[str, list[Image.Image]]] = []
    for row in summary["per_patient"]:
        patient = row["patient"]
        prep = study / "cpu-prep" / patient
        raw = measured_root / patient
        provenance = json.loads((raw / "MEASURED_PIXEL_SOURCE.json").read_text())
        with Image.open(prep / "he.ome.tif") as image:
            he = image.convert("RGB")
        measured = tifffile.imread(raw / "Hoechst.tif")
        store = zarr.open(str(study / "gpu-output" / patient / "gigatime/gigatime_probs.zarr"), mode="r")
        virtual = np.asarray(store["0"][0], dtype=np.float32)
        if he.size != (2048, 2048) or measured.shape != (2048, 2048) or \
                abs(provenance["mpp_x"] - .325) > 1e-6:
            raise ValueError(f"Registered image or physical scale changed: {patient}")
        aligned = zoom(virtual, (2048/virtual.shape[0], 2048/virtual.shape[1]), order=1)
        md, vd = display(measured, virtual=False), display(aligned, virtual=True)
        overlap = np.stack((md, vd, md), axis=-1)
        images = [he, Image.fromarray(md).convert("RGB"),
                  Image.fromarray(vd).convert("RGB"), Image.fromarray(overlap)]
        names = ("he", "measured-Hoechst", "virtual-DAPI", "pixel-overlap")
        for name, image in zip(names, images):
            image.save(output / f"{patient}-{name}.webp", "WEBP", quality=85, method=6)
        contacts.append((patient, images))
        records.append({"patient": patient, "source_mpp": provenance["mpp_x"],
                        "virtual_mpp": .25, "source_window_xy": [provenance["x"], provenance["y"]],
                        "measured_source_url": provenance["source_url"],
                        "pixel_spearman_rho": row["spearman_rho"],
                        "measured_border_core_ratio": row["measured_border_core_ratio"],
                        "virtual_border_core_ratio": row["virtual_border_core_ratio"]})
    for part in range(2):
        selected = contacts[part*5:(part+1)*5]
        width, cell, title = 1600, 400, 45
        sheet = Image.new("RGB", (width, len(selected)*(cell+title)), "white")
        painter = ImageDraw.Draw(sheet)
        for r, (patient, images) in enumerate(selected):
            painter.text((8, r*(cell+title)+7), patient, fill="black")
            for col, image in enumerate(images):
                thumb = image.copy()
                thumb.thumbnail((cell, cell), getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
                sheet.paste(thumb, (col*cell, r*(cell+title)+title))
        sheet.save(output / f"contact-{part+1:02d}.webp", "WEBP", quality=82)
    manifest = {"scope": "Ten reserved ORION QC-passing patients; one registered 2048 px window each",
                "measured": "Raw pixel-level ORION Hoechst channel, not per-cell painted values",
                "virtual": "Native GigaTIME virtual DAPI, resampled to the 0.325 µm/pixel registered source grid",
                "display": "Measured: asinh then 1st–99.7th percentile. Virtual: logit then 1st–99.7th percentile. Display normalization is per image and is excluded from scoring.",
                "overlap": "Magenta measured Hoechst, green virtual DAPI; white indicates co-located bright signal",
                "scores": {key: summary[key] for key in ("equal_patient_median_spearman",
                                                            "equal_patient_median_95pct_bootstrap_ci")},
                "records": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"patients": len(records), "images": len(records)*4,
            "median_rho": summary["equal_patient_median_spearman"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(render(args.study, args.measured, args.scores, args.output)))
