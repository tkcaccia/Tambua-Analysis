#!/usr/bin/env python3
"""CPU-only StarDist detection against patient-separated MoNuSAC masks.

This evaluates detection independently of GigaTIME and cell-type classification.
It writes one atomic checkpoint per ROI and a final cohort marker after all
development and organizer-test ROIs complete. Unmatched predictions are
counted as false positives, with the source-annotation completeness caveat.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This evaluation must explicitly hide all GPUs")

import numpy as np
import tifffile
from stardist.models import StarDist2D


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")
MODEL_DIR = "/home/chiamaka/.keras/models/StarDist2D"
MODEL_NAME = "2D_versatile_he"
PROB_THRESHOLD = 0.48
NMS_THRESHOLD = 0.30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-",
                                     delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        staged = Path(stream.name)
    staged.replace(path)


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected scanner-decoded RGB uint8")
    x = image.astype(np.float32)
    for channel in range(3):
        p1, p998 = np.percentile(x[..., channel], [1, 99.8])
        x[..., channel] = (x[..., channel] - p1) / max(p998 - p1, 1e-6)
    return np.clip(x, 0, 1)


def match_instances(expert: np.ndarray, detected: np.ndarray) -> tuple[list[tuple[int, int, float]], dict]:
    """Greedy one-to-one IoU matching after counting every source instance."""
    if expert.shape != detected.shape:
        raise ValueError("Expert and detected images are on different pixel grids")
    gt = expert.astype(np.int64, copy=False)
    pred = detected.astype(np.int64, copy=False)
    n_pred = int(pred.max()) + 1
    gt_area = np.bincount(gt.ravel())
    pred_area = np.bincount(pred.ravel())
    pairs = gt * n_pred + pred
    keys, counts = np.unique(pairs[(gt > 0) & (pred > 0)], return_counts=True)
    candidates = []
    for key, intersection in zip(keys, counts):
        gi, pi = divmod(int(key), n_pred)
        iou = float(intersection / (gt_area[gi] + pred_area[pi] - intersection))
        if iou >= 0.5:
            candidates.append((gi, pi, iou))
    matched, used_gt, used_pred = [], set(), set()
    for gi, pi, iou in sorted(candidates, key=lambda item: -item[2]):
        if gi not in used_gt and pi not in used_pred:
            matched.append((gi, pi, iou))
            used_gt.add(gi)
            used_pred.add(pi)
    return matched, {"expert_area": gt_area, "detected_area": pred_area}


def score_case(model: StarDist2D, row: dict, prepared: Path, output: Path) -> dict:
    source = prepared / row["sample_id"]
    provenance = json.loads((source / "provenance.json").read_text())
    if any(provenance[key] != row[key] for key in
           ("sample_id", "patient_id", "organ", "split", "width", "height")) or \
            provenance["source_color_reference"] != "scanner_svs_rgb":
        raise ValueError(f"Scanner RGB or source identity mismatch: {row['sample_id']}")
    image = tifffile.imread(source / "he.ome.tif")
    expert = tifffile.imread(source / "expert_nuclei.ome.tif")
    if image.shape != (row["height"], row["width"], 3) or expert.shape != image.shape[:2]:
        raise ValueError(f"Annotation and H&E shape mismatch: {row['sample_id']}")
    with (source / "expert_types.csv").open(newline="", encoding="utf-8") as handle:
        expert_type = {int(item["label_id"]): item["cell_type"]
                       for item in csv.DictReader(handle)}
    expert_ids = set(int(value) for value in np.unique(expert) if value > 0)
    if expert_ids != set(expert_type) or not set(expert_type.values()) <= set(CLASSES):
        raise ValueError(f"Expert mask/label correspondence failed: {row['sample_id']}")
    started = time.monotonic()
    detected, _details = model.predict_instances(
        normalize_rgb(image), prob_thresh=PROB_THRESHOLD,
        nms_thresh=NMS_THRESHOLD, n_tiles=None, show_tile_progress=False)
    elapsed = time.monotonic() - started
    matched, areas = match_instances(expert, detected)
    matched_gt = {item[0] for item in matched}
    detected_ids = set(int(value) for value in np.unique(detected) if value > 0)
    n_gt, n_pred = len(expert_ids), len(detected_ids)
    tp = len(matched)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    counts = Counter(expert_type.values())
    found = Counter(expert_type[gi] for gi in matched_gt)
    oversized = {gi for gi in expert_ids if areas["expert_area"][gi] > 5000}
    result = {
        "sample_id": row["sample_id"], "patient_id": row["patient_id"],
        "organ": row["organ"], "split": "development" if row.get("development") else row["split"],
        "source_mpp": row["mpp"], "width": row["width"], "height": row["height"],
        "source_color_reference": "scanner_svs_rgb",
        "model": MODEL_NAME, "device": "CPU", "prob_threshold": PROB_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD, "expert_instances": n_gt,
        "detected_instances": n_pred, "matched_iou_at_least_0_5": tp,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_matched_iou": float(np.mean([item[2] for item in matched])) if matched else None,
        "expert_by_type": {name: counts[name] for name in CLASSES},
        "matched_by_type": {name: found[name] for name in CLASSES},
        "oversized_expert_regions_gt_5000_px": len(oversized),
        "oversized_expert_regions_matched": len(oversized & matched_gt),
        "inference_seconds": elapsed,
        "he_sha256": digest(source / "he.ome.tif"),
        "expert_mask_sha256": digest(source / "expert_nuclei.ome.tif"),
    }
    output.mkdir(parents=True, exist_ok=False)
    tifffile.imwrite(output / "stardist_labels.tif", detected.astype(np.uint32),
                     compression="zlib")
    atomic_json(output / "CASE_COMPLETE.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--old-prepared", type=Path, required=True)
    ap.add_argument("--new-prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scope", choices=("fit", "development-test"),
                    default="development-test")
    args = ap.parse_args()
    source = json.loads(args.candidate.read_text())
    fold = json.loads(args.fold.read_text())
    if fold["expanded_candidate_sha256"] != digest(args.candidate):
        raise ValueError("Development fold and candidate differ")
    if len(source["records"]) != 213 or args.output.exists():
        raise ValueError("Unexpected candidate or reused detection output")
    prepared_new = json.loads((args.new_prepared / "PREP_COMPLETE.json").read_text())
    if prepared_new["cases"] != 93 or not prepared_new["all_scanner_rgb"]:
        raise ValueError("CPU preparation incomplete")
    new_ids = {r["sample_id"] for r in prepared_new["records"]}
    dev_ids = set(fold["development_patients"])
    if args.scope == "fit":
        selected = [dict(row, development=False) for row in source["records"]
                    if row["split"] == "train" and row["patient_id"] not in dev_ids]
        if len(selected) != 115:
            raise ValueError("Unexpected fit cohort")
    else:
        selected = [dict(row, development=row["patient_id"] in dev_ids)
                    for row in source["records"]
                    if row["split"] == "test" or row["patient_id"] in dev_ids]
        if len(selected) != 98 or sum(row["split"] == "test" for row in selected) != 66:
            raise ValueError("Unexpected development or test cohort")
    args.output.mkdir(parents=True, exist_ok=False)
    model = StarDist2D(None, name=MODEL_NAME, basedir=MODEL_DIR)
    completed = []
    for index, row in enumerate(selected, start=1):
        prepared = args.new_prepared / "samples" if row["sample_id"] in new_ids \
                   else args.old_prepared
        result = score_case(model, row, prepared, args.output / row["sample_id"])
        completed.append(result)
        print(f"{index}/{len(selected)} {result['sample_id']} "
              f"F1={result['f1']:.3f} detected={result['detected_instances']}", flush=True)
    atomic_json(args.output / "COHORT_COMPLETE.json", {
        "scope": args.scope,
        "candidate_sha256": digest(args.candidate), "fold_sha256": digest(args.fold),
        "device": "CPU", "model": MODEL_NAME, "prob_threshold": PROB_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD, "cases": len(completed),
        "records": completed,
        "interpretation": "End-to-end nuclear detection only; no GigaTIME marker quantification or fastPLS typing on detected instances. Source annotations may be incomplete or geometrically implausible.",
    })


if __name__ == "__main__":
    main()
