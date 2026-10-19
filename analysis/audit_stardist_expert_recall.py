#!/usr/bin/env python3
"""CPU-only StarDist recall on annotated MoNuSAC nuclei, canonical scanner RGB.

MoNuSAC does not annotate every visible nucleus: only sensitivity over its
annotated instances is valid here, not whole-image detection precision.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def annotated_recall(pred: np.ndarray, expert: np.ndarray) -> dict:
    if pred.shape != expert.shape:
        raise ValueError("Expert and StarDist masks must share a pixel grid")
    expert_ids, expert_areas = np.unique(expert[expert > 0], return_counts=True)
    pred_ids, pred_areas = np.unique(pred[pred > 0], return_counts=True)
    if not len(expert_ids):
        raise ValueError("No annotated expert nuclei")
    exp_area = dict(zip(expert_ids.tolist(), expert_areas.tolist()))
    det_area = dict(zip(pred_ids.tolist(), pred_areas.tolist()))
    both = (expert > 0) & (pred > 0)
    pairs = (expert[both].astype(np.int64) << 32) | pred[both].astype(np.int64)
    keys, areas = np.unique(pairs, return_counts=True)
    best_iou = {int(k): 0.0 for k in expert_ids}
    best_cover = {int(k): 0.0 for k in expert_ids}
    for key, overlap in zip(keys, areas):
        e, d = int(key >> 32), int(key & ((1 << 32) - 1))
        iou = overlap / (exp_area[e] + det_area[d] - overlap)
        best_iou[e] = max(best_iou[e], iou)
        best_cover[e] = max(best_cover[e], overlap / exp_area[e])
    return {
        "annotated_nuclei": len(expert_ids),
        "predicted_instances": len(pred_ids),
        "recall_iou_0_3": sum(v >= .3 for v in best_iou.values()) / len(expert_ids),
        "recall_iou_0_5": sum(v >= .5 for v in best_iou.values()) / len(expert_ids),
        "recall_expert_coverage_0_5": sum(v >= .5 for v in best_cover.values()) / len(expert_ids),
        "median_best_iou": statistics.median(best_iou.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="test", choices=("train", "test"))
    p.add_argument("--prob", type=float, nargs="+", default=[.4, .52, .6924782541382084])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((args.dataset / "COHORT.json").read_text())["records"]
    cases = sorted((r for r in registry if r["split"] == args.split), key=lambda r: r["sample_id"])
    model = StarDist2D.from_pretrained("2D_versatile_he")
    rows = []
    for record in cases:
        case = record["sample_id"]
        folder = args.dataset / case
        he = tifffile.imread(folder / "he.ome.tif")
        expert = tifffile.imread(folder / "expert_nuclei.ome.tif")
        if he.shape[:2] != expert.shape or he.shape[2] != 3:
            raise ValueError(f"Mismatched source: {case}")
        size = (round(he.shape[1] * record["mpp"] / .25),
                round(he.shape[0] * record["mpp"] / .25))
        rgb = np.asarray(Image.fromarray(he).resize(size, Image.Resampling.BICUBIC))
        start = time.monotonic()
        prob, dist = model.predict(normalize(rgb), n_tiles=(2, 2, 1), show_tile_progress=False)
        inference_s = time.monotonic() - start
        for threshold in args.prob:
            nms = .28 if threshold == .52 else .3
            labels, _ = model._instances_from_prediction(
                rgb.shape[:2], prob, dist, prob_thresh=threshold, nms_thresh=nms)
            native = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(
                (he.shape[1], he.shape[0]), Image.Resampling.NEAREST))
            metrics = annotated_recall(native, expert)
            row = {"case": case, "patient": record["patient_id"], "organ": record["organ"],
                   "split": args.split, "mpp": record["mpp"], "prob_thresh": threshold,
                   "nms_thresh": nms, "inference_seconds": inference_s, **metrics}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (args.output / f"{args.split}_expert_recall.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
