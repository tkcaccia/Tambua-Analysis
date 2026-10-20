#!/usr/bin/env python3
"""Render every saved spatial GigaTIME channel for the corrected QC cohort.

This is CPU-only and never runs GigaTIME. Native 16-bit model-grid values are
read from the existing Zarr store; logit/percentile scaling affects display
copies only. The manifest preserves the original quantiles and resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import zarr


CHANNELS = (
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
    "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
    "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B",
    "Transgelin",
)


def display(raw: np.ndarray) -> np.ndarray:
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError("Expected a two-dimensional native uint16 GigaTIME channel")
    score = np.clip(raw.astype(np.float32) / 65535.0, 1e-4, 1 - 1e-4)
    transformed = np.log(score / (1 - score))
    low, high = np.quantile(transformed, [.01, .997])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("GigaTIME channel has no usable display contrast")
    return np.rint(np.clip((transformed - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def render(source: Path, qc_summary: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Rendering must explicitly hide all GPUs")
    qc = json.loads(qc_summary.read_text())
    if qc.get("cases") != 120 or qc.get("status") != "complete":
        raise ValueError("Corrected StarDist-reference QC summary is incomplete")
    selected = [row for row in qc["records"] if
                all(value == "pass" for value in row["stardist_nuclear_qc"].values())]
    if len(selected) != 47 or output.exists():
        raise ValueError("Frozen 47-image QC cohort changed or output exists")
    output.mkdir(parents=True)
    records = []
    for row in selected:
        sample = row["sample_id"]
        case = source / "output" / sample
        complete = json.loads((case / "VALIDATION.json").read_text())
        if complete.get("source_color_reference") != "scanner_svs_rgb":
            raise ValueError(f"Wrong source color order: {sample}")
        seam = json.loads((case / "gigatime_output/gigatime_seam_qc.json").read_text())
        marker_qc = json.loads((case / "gigatime_output/gigatime_marker_score_qc.json").read_text())
        if seam.get("status") != "pass" or marker_qc.get("status") not in ("pass", "review"):
            raise ValueError(f"All-channel seam or marker-distribution QC failed: {sample}")
        store = zarr.open(str(case / "gigatime_output/gigatime_probs.zarr"), mode="r")
        meta = dict(store.attrs)["gigatime"]
        if tuple(meta["store_channels"]) != CHANNELS or \
                abs(meta["effective_mpp"] - .25) > 1e-6:
            raise ValueError(f"Marker panel or model grid differs: {sample}")
        array = store["0"]
        if array.ndim != 3 or array.shape[0] != len(CHANNELS) or \
                array.dtype != np.uint16:
            raise ValueError(f"Native marker array differs: {sample}")
        quantiles = {}
        for index, channel in enumerate(CHANNELS):
            raw = np.asarray(array[index])
            image = Image.fromarray(display(raw), "L")
            name = f"{sample}--{index:02d}.webp"
            image.save(output / name, "WEBP", quality=78, method=4)
            quantiles[channel] = [float(value) for value in np.quantile(raw, [.01, .5, .997])]
        records.append({"sample_id": sample, "split": row["split"],
                        "source_mpp": meta["source_mpp"], "model_mpp": .25,
                        "model_shape_yx": list(array.shape[1:]),
                        "all_channel_seam_qc": seam["status"],
                        "seam_channel_boundary_tests": seam["channel_boundary_tests"],
                        "marker_distribution_qc": marker_qc["status"],
                        "marker_distribution_warnings": marker_qc.get("warnings", []),
                        "native_uint16_quantiles_p01_p50_p997": quantiles})
    manifest = {"status": "complete", "device": "CPU", "cases": len(records),
                "channel_names": list(CHANNELS),
                "model_grid_mpp": .25,
                "display": "Per-image, per-channel logit of native uint16/65535, then 1st–99.7th percentile to 8-bit. Display-only; not suitable for cross-image quantitative comparison.",
                "native_values": "Uncalibrated GigaTIME uint16 model scores; the original Zarr stores remain on Chiamaka.",
                "qc_limit": "DAPI alignment and spatial-shadow QC passes for all images. All-channel seam QC passes. Distribution-review warnings flag near-flat channels and do not assert biological validity.",
                "qc_summary_sha256": hashlib.sha256(qc_summary.read_bytes()).hexdigest(),
                "records": records}
    staged = output / "manifest.json.tmp"
    staged.write_text(json.dumps(manifest, indent=2) + "\n")
    staged.replace(output / "manifest.json")
    return {"cases": len(records), "channels": len(CHANNELS),
            "images": len(records) * len(CHANNELS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qc-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.qc_summary, args.output)))
