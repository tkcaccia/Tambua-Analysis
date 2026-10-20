#!/usr/bin/env python3
"""Read-only RGB/stain audit of completed MoNuSAC GigaTIME cases.

This is diagnostic aggregation, not model inference or QC-threshold tuning.
It does not change the running cohort, pass labels, or input images.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile


def sample_color(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB:
            raise ValueError(f"Source is not tagged RGB: {path}")
        rgb = tif.asarray()
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB H&E: {path}")
    pixels = rgb[::3, ::3].reshape(-1, 3)
    tissue = (pixels.mean(axis=1) < 240) & (np.ptp(pixels, axis=1) > 12)
    if int(tissue.sum()) < 100:
        raise ValueError(f"Insufficient tissue pixels: {path}")
    selected = pixels[tissue]
    lab = cv2.cvtColor(selected[None], cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return {
        "sampled_pixels": int(len(pixels)),
        "sampled_tissue_pixels": int(len(selected)),
        "sampled_tissue_fraction": round(float(tissue.mean()), 6),
        "median_rgb": [float(v) for v in np.median(selected, axis=0)],
        "median_lab_opencv_uint8": [float(v) for v in np.median(lab, axis=0)],
    }


def grouped_summary(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["organ"], row["group"])].append(row)
    summary = []
    for (organ, group), rows in sorted(groups.items()):
        lab = np.array([r["color"]["median_lab_opencv_uint8"] for r in rows])
        rgb = np.array([r["color"]["median_rgb"] for r in rows])
        summary.append({"organ": organ, "group": group, "images": len(rows),
                        "across_image_median_lab": [float(v) for v in np.median(lab, axis=0)],
                        "across_image_median_rgb": [float(v) for v in np.median(rgb, axis=0)]})
    return summary


def run(old_summary: Path, old_rgb_root: Path, new_rgb_root: Path,
        new_output_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    original = json.loads(old_summary.read_text())
    records = []
    for item in original["records"]:
        sample = item["sample_id"]
        status = item["stardist_nuclear_qc"]["status"]
        if status not in {"pass", "fail"}:
            raise ValueError(f"Unexpected old QC status: {sample}: {status}")
        records.append({"sample_id": sample, "organ": item["organ"],
                        "group": f"prior_{status}",
                        "color": sample_color(old_rgb_root / sample / "he.ome.tif")})
    completed = sorted(new_output_root.glob("*/CASE_COMPLETE.json"))
    if not completed:
        raise ValueError("No completed expansion cases")
    for marker in completed:
        item = json.loads(marker.read_text())
        sample = item["sample_id"]
        if sample != marker.parent.name:
            raise ValueError(f"Case name mismatch: {marker}")
        qc = json.loads((marker.parent / "stardist_qc.json").read_text())
        records.append({"sample_id": sample, "organ": item["organ"],
                        "group": "expanded_complete", "qc_status": qc["status"],
                        "auc": qc["auc"],
                        "nuclear_boundary_contrast": qc["nuclear_boundary_contrast"],
                        "color": sample_color(new_rgb_root / sample / "he.ome.tif")})
    report = {
        "status": "partial_read_only_audit",
        "prior_images": len(original["records"]),
        "expanded_completed_images": len(completed),
        "color_method": "Every third RGB pixel; tissue if mean RGB<240 and RGB channel range>12; per-image tissue-pixel median, then per-organ/group median. LAB values use OpenCV uint8 encoding.",
        "causal_limit": "Brightness and stain associations cannot establish a cause of QC failure. No threshold, image, or prediction was changed.",
        "groups": grouped_summary(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--old-rgb-root", type=Path, required=True)
    parser.add_argument("--new-rgb-root", type=Path, required=True)
    parser.add_argument("--new-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.old_summary, args.old_rgb_root, args.new_rgb_root,
                 args.new_output_root, args.output)
    print(json.dumps({"prior": result["prior_images"],
                      "expanded_complete": result["expanded_completed_images"],
                      "groups": result["groups"]}))
