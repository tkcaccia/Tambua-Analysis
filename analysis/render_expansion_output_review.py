#!/usr/bin/env python3
"""Render completed MoNuSAC expansion outputs for human image-level QC.

This read-only audit accepts a partial CPU inference bundle. A contact sheet is
evidence for visual inspection, never a substitute for a QC pass or a claim
that every image in the frozen expansion has been reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import zarr

from render_all_monusac_markers import CHANNELS, display


PANELS = ("Scanner RGB", "Virtual DAPI", "DAPI / StarDist", "CK", "CD68")
TILE = 250
HEADER = 54
PAGE_SIZE = 6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(image: Image.Image) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((TILE - 8, TILE - 8), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (TILE, TILE), "white")
    tile.paste(copy, ((TILE - copy.width) // 2, (TILE - copy.height) // 2))
    return tile


def case_panels(prepared: Path, inference: Path, row: dict) -> tuple[list[Image.Image], dict]:
    sample = row["sample_id"]
    case = inference / sample
    saved = json.loads((case / "CASE_COMPLETE.json").read_text())
    rgb_path = prepared / "samples" / sample / "he.ome.tif"
    if row["source_color_reference"] != "scanner_svs_rgb" or \
            saved["sample_id"] != sample or saved["source_rgb_sha256"] != sha(rgb_path) or \
            abs(saved["source_mpp"] - row["mpp"]) > 1e-6 or \
            abs(saved["model_mpp"] - .25) > 1e-6 or \
            saved["seam_qc"] != "pass" or tuple(saved["channels"]) != CHANNELS:
        raise ValueError(f"Output identity, calibration or seam check failed: {sample}")
    with Image.open(rgb_path) as opened:
        if opened.mode != "RGB" or opened.size != (row["width"], row["height"]):
            raise ValueError(f"Scanner RGB image shape changed: {sample}")
        rgb = opened.copy()
    store = zarr.open(str(case / "gigatime_output/gigatime_probs.zarr"), mode="r")
    meta = store.attrs["gigatime"]
    array = store["0"]
    if tuple(meta["store_channels"]) != CHANNELS or \
            abs(meta["source_mpp"] - row["mpp"]) > 1e-6 or \
            abs(meta["effective_mpp"] - .25) > 1e-6 or \
            array.shape[0] != 23 or array.dtype != np.uint16:
        raise ValueError(f"Native GigaTIME store changed: {sample}")
    marker_panels = []
    quantiles = {}
    for channel in ("DAPI", "CK", "CD68"):
        raw = np.asarray(array[CHANNELS.index(channel)])
        quantiles[channel] = [float(v) for v in np.quantile(raw, [.01, .5, .997])]
        try:
            marker_panels.append(Image.fromarray(display(raw), mode="L"))
        except ValueError:
            marker_panels.append(Image.new("L", (raw.shape[1], raw.shape[0]), 0))
            quantiles[channel].append("flat-channel-display")
    with Image.open(case / "stardist_qc.png") as opened:
        overlay = opened.copy()
    qc = json.loads((case / "stardist_qc.json").read_text())
    if qc["status"] != saved["nuclear_qc"]["status"]:
        raise ValueError(f"Nuclear QC status changed: {sample}")
    images = [rgb, marker_panels[0], overlay, marker_panels[1], marker_panels[2]]
    result = {"sample_id": sample, "patient_id": row["patient_id"],
              "organ": row["organ"], "split": row["split"],
              "source_mpp": row["mpp"], "model_mpp": .25,
              "qc_status": qc["status"], "alignment_status": qc["alignment_status"],
              "border_shadow_status": qc["border_shadow_status"],
              "spatial_shadow_status": qc["spatial_shadow_status"],
              "dapi_auc": qc["auc"],
              "dapi_boundary_contrast": qc["nuclear_boundary_contrast"],
              "raw_quantiles_p01_p50_p997": quantiles,
              "visual_review_status": "pending"}
    return images, result


def render(prepared: Path, inference: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Visual-audit renderer must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    prep = json.loads((prepared / "PREP_COMPLETE.json").read_text())
    identity = json.loads((inference / "BUNDLE_IDENTITY.json").read_text())
    if prep["cases"] != 93 or not prep["all_scanner_rgb"] or \
            identity["device"] != "CPU" or \
            identity["prepared_sha256"] != sha(prepared / "PREP_COMPLETE.json") or \
            set(identity["case_order"]) != {row["sample_id"] for row in prep["records"]}:
        raise ValueError("Frozen CPU expansion identity changed")
    source = {row["sample_id"]: row for row in prep["records"]}
    complete = [sample for sample in identity["case_order"]
                if (inference / sample / "CASE_COMPLETE.json").is_file()]
    if not complete:
        raise ValueError("No completed CPU cases available for visual review")
    output.mkdir(parents=True)
    records = []
    pages = []
    for page_no, offset in enumerate(range(0, len(complete), PAGE_SIZE), 1):
        batch = complete[offset:offset + PAGE_SIZE]
        canvas = Image.new("RGB", (TILE * len(PANELS),
                                   HEADER + len(batch) * (TILE + HEADER)), "white")
        pen = ImageDraw.Draw(canvas)
        for column, name in enumerate(PANELS):
            pen.text((column * TILE + 7, 15), name, fill="black")
        for index, sample in enumerate(batch):
            images, record = case_panels(prepared, inference, source[sample])
            records.append(record)
            y = HEADER + index * (TILE + HEADER)
            detail = (f"{sample} | {record['organ']} {record['split']} | "
                      f"QC {record['qc_status']} AUC {record['dapi_auc']:.3f} "
                      f"contrast {record['dapi_boundary_contrast']:.3f}")
            pen.text((7, y + 3), detail, fill="black")
            for column, picture in enumerate(images):
                canvas.paste(fit(picture), (column * TILE, y + HEADER))
        filename = f"output-qc-{page_no:02d}.webp"
        canvas.save(output / filename, "WEBP", quality=88, method=4)
        pages.append({"file": filename, "sample_ids": batch})
    bundle_marker = inference / "BUNDLE_COMPLETE.json"
    bundle = json.loads(bundle_marker.read_text()) if bundle_marker.is_file() else {}
    bundle_done = bundle.get("status") == "complete" and bundle.get("cases") == 93
    report = {"status": "complete" if len(complete) == 93 and bundle_done else "partial",
              "completed_cases_rendered": len(complete), "expected_cases": 93,
              "qc_status_counts": dict(Counter(row["qc_status"] for row in records)),
              "visual_review_status": "pending for every rendered case",
              "panels": PANELS, "pages": pages, "records": records,
              "display": "Marker channels use per-image logit/1st–99.7th percentile display scaling; original uint16 scores remain unchanged. These previews cannot establish marker validity alone.",
              "limit": "A partial bundle is diagnostic only. QC-failing cases are excluded from classifier analysis; human visual review is separately required before publication."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.prepared, args.inference, args.output)
    print(json.dumps({key: result[key] for key in
                      ("status", "completed_cases_rendered", "qc_status_counts")}, indent=2))
