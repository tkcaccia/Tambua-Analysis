#!/usr/bin/env python3
"""Add label-free H&E nuclear morphology to saved GigaTIME cell features.

Development audit only. Features come from the scanner-RGB image and independent
StarDist instance mask. Expert cell types are carried through but never used to
compute morphology. Every original GigaTIME channel remains in the output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


FEATURES = ("morph_area_um2", "morph_perimeter_um", "morph_elongation",
            "morph_compactness", "morph_od_red_mean", "morph_od_green_mean",
            "morph_od_blue_mean", "morph_od_red_sd", "morph_od_green_sd",
            "morph_od_blue_sd")


def image_features(rgb_path: Path, mask_path: Path, provenance_path: Path) -> dict[int, dict]:
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("source_color_reference") != "scanner_svs_rgb":
        raise ValueError(f"Source color order is not scanner RGB: {rgb_path}")
    with Image.open(rgb_path) as handle:
        if handle.mode != "RGB":
            raise ValueError("Expected a real scanner-RGB image")
        rgb = np.asarray(handle, dtype=np.uint8)
    mask = tifffile.imread(mask_path).astype(np.int64, copy=False)
    if rgb.shape[:2] != mask.shape or mask.min() < 0:
        raise ValueError("H&E and StarDist pixel grids differ")
    mpp = float(provenance["mpp"])
    if not .15 <= mpp <= .4:
        raise ValueError("Unexpected physical pixel spacing")
    valid = mask > 0
    ids = mask[valid]
    count = np.bincount(ids, minlength=int(mask.max())+1).astype(float)
    yy, xx = np.nonzero(valid)
    xmean = np.bincount(ids, weights=xx, minlength=len(count)) / np.maximum(count, 1)
    ymean = np.bincount(ids, weights=yy, minlength=len(count)) / np.maximum(count, 1)
    dx, dy = xx - xmean[ids], yy - ymean[ids]
    var_x = np.bincount(ids, weights=dx*dx, minlength=len(count)) / np.maximum(count, 1)
    var_y = np.bincount(ids, weights=dy*dy, minlength=len(count)) / np.maximum(count, 1)
    cov = np.bincount(ids, weights=dx*dy, minlength=len(count)) / np.maximum(count, 1)
    radial = np.sqrt(np.maximum(0, (var_x-var_y)**2 + 4*cov**2))
    major = np.maximum((var_x+var_y+radial)/2, .01)
    minor = np.maximum((var_x+var_y-radial)/2, .01)
    elongation = np.sqrt(major/minor)

    # Four-connected boundary length, counting both sides of touching labels.
    edge = np.zeros(mask.shape, dtype=np.uint8)
    edge[0, :] += 1
    edge[-1, :] += 1
    edge[:, 0] += 1
    edge[:, -1] += 1
    diff = mask[1:, :] != mask[:-1, :]
    edge[1:, :] += diff
    edge[:-1, :] += diff
    diff = mask[:, 1:] != mask[:, :-1]
    edge[:, 1:] += diff
    edge[:, :-1] += diff
    perimeter = np.bincount(ids, weights=edge[valid], minlength=len(count))
    od = -np.log((rgb[valid].astype(float)+1)/256)
    means, sds = [], []
    for channel in range(3):
        mean = np.bincount(ids, weights=od[:, channel], minlength=len(count)) / np.maximum(count, 1)
        second = np.bincount(ids, weights=od[:, channel]**2, minlength=len(count)) / np.maximum(count, 1)
        means.append(mean)
        sds.append(np.sqrt(np.maximum(0, second-mean**2)))
    return {label: dict(zip(FEATURES,
                            (float(count[label]*mpp*mpp), float(perimeter[label]*mpp),
                             float(elongation[label]),
                             float(perimeter[label]**2/(4*np.pi*count[label])),
                             *(float(column[label]) for column in means),
                             *(float(column[label]) for column in sds)), strict=True))
            for label in range(1, len(count)) if count[label] > 0}


def run(table: Path, rgb_root: Path, detected_root: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Morphology audit must explicitly hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        original_columns = reader.fieldnames or []
        records = [row for row in reader if row["split"] == "train"]
    marker_columns = [name for name in original_columns
                      if name.startswith("nuclei_") or
                      (name.startswith("cell_") and name != "cell_type")]
    if len(records) != 10049 or len(marker_columns) != 46:
        raise ValueError("Saved fit-cell cohort or 23-channel panel changed")
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_sample[row["sample_id"]].append(row)
    if len(by_sample) != 33:
        raise ValueError("Saved fit-image set changed")
    enriched = []
    for sample in sorted(by_sample):
        folder = rgb_root / sample
        found = image_features(folder / "he.ome.tif",
                               detected_root / sample / "stardist_labels.tif",
                               folder / "provenance.json")
        for row in by_sample[sample]:
            label = int(row["label_id"])
            if label not in found:
                raise ValueError(f"Matched cell has no independent StarDist nucleus: {sample}/{label}")
            enriched.append({**row, **found[label]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=original_columns + list(FEATURES))
        writer.writeheader()
        writer.writerows(enriched)
    result = {"status": "complete", "device": "CPU", "fit_images": len(by_sample),
              "fit_cells": len(enriched), "marker_features_preserved": len(marker_columns),
              "morphology_features_added": FEATURES,
              "original_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
              "output_table_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "scope": "Label-free scanner-RGB H&E and StarDist morphology appended to all-channel fit-cell table",
              "limit": "No new biological examples or independent test result; morphology utility must be assessed by patient-held-out evaluation."}
    Path(str(output)+".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--rgb-root", type=Path, required=True)
    ap.add_argument("--detected-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.table, args.rgb_root, args.detected_root, args.output)))
