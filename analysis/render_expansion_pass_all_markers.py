#!/usr/bin/env python3
"""Render all 23 native spatial GigaTIME channels for one expansion QC pass.

Read-only CPU visualization. No marker scores or cell labels are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tifffile
import zarr

from render_all_monusac_markers import CHANNELS, display


SIDE = 230
LABEL = 31
COLUMNS = 5


def tile(image: Image.Image) -> Image.Image:
    work = image.convert("RGB")
    work.thumbnail((SIDE - 10, SIDE - 10), Image.Resampling.LANCZOS)
    return ImageOps.pad(work, (SIDE, SIDE), color="white")


def render(prepared: Path, inference: Path, sample: str, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("All-marker visual review must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    case = inference / sample
    complete = json.loads((case / "CASE_COMPLETE.json").read_text())
    qc = json.loads((case / "stardist_qc.json").read_text())
    seam = json.loads((case / "gigatime_output" / "gigatime_seam_qc.json").read_text())
    score = json.loads((case / "gigatime_output" / "gigatime_marker_score_qc.json").read_text())
    source = prepared / "samples" / sample / "he.ome.tif"
    if sample != complete["sample_id"] or qc["status"] != "pass" or \
            complete["nuclear_qc"]["status"] != "pass" or seam["status"] != "pass" or \
            complete["seam_qc"] != "pass" or score["status"] not in {"pass", "review"} or \
            complete["source_rgb_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError(f"Identity, input RGB, nuclear QC or seam QC failed: {sample}")
    with tifffile.TiffFile(source) as tif:
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB:
            raise ValueError("Source OME-TIFF is not tagged RGB")
        rgb = tif.asarray()
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Source is not uint8 scanner RGB")
    store = zarr.open(str(case / "gigatime_output" / "gigatime_probs.zarr"), mode="r")
    meta = store.attrs["gigatime"]
    data = store["0"]
    if tuple(meta["store_channels"]) != CHANNELS or \
            abs(float(meta["effective_mpp"]) - .25) > 1e-6 or \
            abs(float(meta["source_mpp"]) - float(complete["source_mpp"])) > 1e-6 or \
            data.shape[0] != len(CHANNELS) or data.dtype != np.uint16:
        raise ValueError("Marker identity or physical scale changed")
    with Image.open(case / "stardist_qc.png") as shown:
        overlay = shown.copy()
    panels = [("Scanner RGB H&E", Image.fromarray(rgb)),
              ("DAPI / StarDist", overlay)]
    quantiles = {}
    for index, name in enumerate(CHANNELS):
        raw = np.asarray(data[index])
        quantiles[name] = [float(v) for v in np.quantile(raw, [.01, .5, .997])]
        try:
            visual = display(raw)
        except ValueError:
            visual = np.zeros(raw.shape, dtype=np.uint8)
            quantiles[name].append("flat-channel-display")
        panels.append((name, Image.fromarray(visual, "L")))
    rows = (len(panels) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * SIDE, rows * (SIDE + LABEL)), "white")
    pen = ImageDraw.Draw(sheet)
    for index, (name, picture) in enumerate(panels):
        x, y = (index % COLUMNS) * SIDE, (index // COLUMNS) * (SIDE + LABEL)
        pen.text((x + 6, y + 7), name, fill="black")
        sheet.paste(tile(picture), (x, y + LABEL))
    output.mkdir(parents=True)
    sheet.save(output / "all-23-markers.webp", "WEBP", quality=88, method=4)
    report = {"sample_id": sample, "status": "awaiting_manual_visual_review",
              "source_rgb_sha256": complete["source_rgb_sha256"],
              "source_mpp": complete["source_mpp"], "model_mpp": meta["effective_mpp"],
              "model_shape_yx": list(data.shape[1:]), "nuclear_qc": qc,
              "all_channel_seam_qc": seam["status"],
              "marker_distribution_qc": score["status"],
              "marker_distribution_warnings": score.get("warnings", []),
              "native_uint16_quantiles_p01_p50_p997": quantiles,
              "display": "Each marker is independently logit/1st–99.7th percentile stretched for visual review only; native uint16 scores remain unchanged.",
              "limit": "DAPI technical QC and all-channel seam QC passed; this is not protein-specific biological validation."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.prepared, args.inference, args.sample, args.output)
    print(json.dumps({"sample_id": report["sample_id"],
                      "status": report["status"],
                      "marker_distribution_qc": report["marker_distribution_qc"]}))
