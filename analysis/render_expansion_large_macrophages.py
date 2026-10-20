#!/usr/bin/env python3
"""Render oversized expert macrophage regions against real H&E and StarDist.

CPU-only visual audit. It never changes labels, masks or the QC cohort.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import tifffile


def run(prepared: Path, fit_match: Path, eval_match: Path,
        fit_detected: Path, eval_detected: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Large-region visual audit must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    selected = {row["sample_id"] for row in
                json.loads((prepared / "PREP_COMPLETE.json").read_text())["records"]}
    candidates = []
    for folder in (fit_match, eval_match):
        for path in folder.glob("TCGA-*.json"):
            sample = path.stem
            if sample not in selected:
                continue
            case = json.loads(path.read_text())
            source = prepared / "samples" / sample
            with Image.open(source / "he.ome.tif") as handle:
                if handle.mode != "RGB":
                    raise ValueError("Expected scanner RGB")
                rgb = np.asarray(handle)
            expert = tifffile.imread(source / "expert_nuclei.ome.tif")
            detected_root = fit_detected if case["split"] == "train" else eval_detected
            detected = tifffile.imread(detected_root / sample / "stardist_labels.tif")
            if rgb.shape[:2] != expert.shape or expert.shape != detected.shape:
                raise ValueError(f"Physical image grid mismatch: {sample}")
            areas = np.bincount(expert.ravel())
            for row in case["assignments"]:
                if row["cell_type"] != "macrophage":
                    continue
                area = int(areas[row["expert_id"]])
                if area <= 5000:
                    continue
                expert_region = expert == row["expert_id"]
                detected_region = detected == row["detected_id"]
                canvas = rgb.copy()
                canvas[expert_region] = (.55 * canvas[expert_region] +
                                         .45 * np.array([255, 0, 170])).astype(np.uint8)
                outline = detected_region & ~binary_erosion(detected_region)
                canvas[outline] = [0, 255, 255]
                image = Image.fromarray(canvas)
                image.thumbnail((300, 300), Image.LANCZOS)
                candidates.append((sample, case["split"], case["organ"], area,
                                   float(row["pred_contained"]), image))
    candidates.sort(key=lambda row: -row[3])
    if len(candidates) != 40:
        raise ValueError(f"Expected 40 matched oversized macrophage regions, got {len(candidates)}")
    output.mkdir(parents=True)
    manifest = []
    for page in range(4):
        sheet = Image.new("RGB", (1600, 680), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (sample, split, organ, area, contained, image) in enumerate(
                candidates[page * 10:(page + 1) * 10]):
            x, y = (index % 5) * 320, (index // 5) * 340
            draw.text((x + 4, y + 2), f"{page * 10 + index + 1}. {sample}", fill="black")
            draw.text((x + 4, y + 22),
                      f"{split}/{organ} area={area} contained={contained:.2f}", fill="black")
            sheet.paste(image, (x + 4, y + 42))
            manifest.append({"rank": page * 10 + index + 1, "sample_id": sample,
                             "split": split, "organ": organ, "expert_area_native_px": area,
                             "detected_nucleus_fraction_in_expert_region": contained,
                             "sheet": f"large-macrophage-{page + 1}.jpg"})
        sheet.save(output / f"large-macrophage-{page + 1}.jpg", quality=93)
    result = {"status": "visual_review_pending", "device": "CPU", "regions": len(manifest),
              "overlay": "Magenta expert macrophage region; cyan matched StarDist nucleus outline; scanner RGB H&E background",
              "limit": "These source annotations may cover cytoplasm or structures larger than one cell; visual inspection is required before use as reliable cell-type labels.",
              "records": manifest}
    (output / "MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "fit-match", "eval-match", "fit-detected", "eval-detected", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.prepared, args.fit_match, args.eval_match,
                         args.fit_detected, args.eval_detected, args.output)))
