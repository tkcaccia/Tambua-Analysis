#!/usr/bin/env python3
"""Map source expert cell types to StarDist nuclei without changing labels.

MoNuSAC macrophage polygons often cover the whole cell, unlike small nuclear
polygons for other types. Ordinary nuclear IoU matching therefore misses them.
This audit accepts a macrophage polygon only when exactly one detected nucleus
lies at least 75% inside it, after nuclear IoU matches have been assigned.
Ambiguous polygons are withheld rather than given a guessed label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import tifffile


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def overlap_pairs(expert: np.ndarray, detected: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if expert.shape != detected.shape:
        raise ValueError("Expert and detected masks differ in shape")
    gt = expert.astype(np.int64, copy=False)
    pr = detected.astype(np.int64, copy=False)
    gt_area = np.bincount(gt.ravel())
    pred_area = np.bincount(pr.ravel())
    base = len(pred_area)
    keys, counts = np.unique((gt * base + pr)[(gt > 0) & (pr > 0)], return_counts=True)
    info = {}
    for key, count in zip(keys, counts):
        gi, pi = divmod(int(key), base)
        info[(gi, pi)] = {"intersection": int(count),
                          "iou": float(count / (gt_area[gi] + pred_area[pi] - count)),
                          "pred_contained": float(count / pred_area[pi])}
    return gt_area, pred_area, info


def assign(expert: np.ndarray, detected: np.ndarray,
           labels: dict[int, str]) -> tuple[list[dict], dict]:
    gt_area, pred_area, info = overlap_pairs(expert, detected)
    ids = set(int(x) for x in np.unique(expert) if x > 0)
    if ids != set(labels):
        raise ValueError("Source class table does not match expert mask IDs")
    candidates = [(gi, pi, pair["iou"]) for (gi, pi), pair in info.items()
                  if pair["iou"] >= 0.5]
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    accepted: list[dict] = []
    for gi, pi, iou in sorted(candidates, key=lambda item: -item[2]):
        if gi in used_gt or pi in used_pred:
            continue
        accepted.append({"expert_id": gi, "detected_id": pi,
                         "cell_type": labels[gi], "rule": "nuclear_iou",
                         "iou": iou, "pred_contained": info[(gi, pi)]["pred_contained"]})
        used_gt.add(gi)
        used_pred.add(pi)
    ambiguous = []
    for gi in sorted(ids - used_gt):
        if labels[gi] != "macrophage":
            continue
        possible = [(pi, pair["pred_contained"], pair["iou"])
                    for (other_gt, pi), pair in info.items()
                    if other_gt == gi and pi not in used_pred and
                    pair["pred_contained"] >= 0.75]
        if len(possible) == 1:
            pi, contained, iou = possible[0]
            accepted.append({"expert_id": gi, "detected_id": pi,
                             "cell_type": "macrophage", "rule": "unique_contained_nucleus",
                             "iou": iou, "pred_contained": contained})
            used_gt.add(gi)
            used_pred.add(pi)
        elif len(possible) > 1:
            ambiguous.append(gi)
    if len({a["expert_id"] for a in accepted}) != len(accepted) or \
            len({a["detected_id"] for a in accepted}) != len(accepted):
        raise ValueError("Assignment is not one-to-one")
    totals = Counter(labels.values())
    found = Counter(a["cell_type"] for a in accepted)
    details = {"expert_by_type": {name: totals[name] for name in CLASSES},
               "matched_by_type": {name: found[name] for name in CLASSES},
               "macrophage_nuclear_iou": sum(a["cell_type"] == "macrophage" and
                                               a["rule"] == "nuclear_iou" for a in accepted),
               "macrophage_unique_containment": sum(a["rule"] == "unique_contained_nucleus"
                                                    for a in accepted),
               "ambiguous_macrophage_regions": len(ambiguous),
               "unmatched_expert_regions": len(ids - used_gt),
               "unmatched_detected_instances": len(set(int(x) for x in
                                                       np.unique(detected) if x > 0) - used_pred),
               "expert_pixels_by_class": {name: int(sum(gt_area[gi] for gi in ids
                                                        if labels[gi] == name)) for name in CLASSES}}
    return accepted, details


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detected-root", type=Path, required=True)
    ap.add_argument("--old-prepared", type=Path, required=True)
    ap.add_argument("--new-prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cohort = json.loads((args.detected_root / "COHORT_COMPLETE.json").read_text())
    new_ids = {r["sample_id"] for r in json.loads(
        (args.new_prepared / "PREP_COMPLETE.json").read_text())["records"]}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summary = []
    for row in cohort["records"]:
        sample = row["sample_id"]
        prepared = args.new_prepared / "samples" if sample in new_ids else args.old_prepared
        gt = tifffile.imread(prepared / sample / "expert_nuclei.ome.tif")
        pr = tifffile.imread(args.detected_root / sample / "stardist_labels.tif")
        with (prepared / sample / "expert_types.csv").open(newline="") as stream:
            classes = {int(entry["label_id"]): entry["cell_type"]
                       for entry in csv.DictReader(stream)}
        pairs, details = assign(gt, pr, classes)
        case = {"sample_id": sample, "patient_id": row["patient_id"],
                "organ": row["organ"], "split": row["split"],
                **details, "assignments": pairs}
        (args.output / f"{sample}.json").write_text(json.dumps(case, indent=2) + "\n")
        summary.append({key: case[key] for key in case if key != "assignments"})
    (args.output / "SUMMARY.json").write_text(json.dumps({"scope": cohort["scope"],
        "rule": "IoU>=0.5 or unique macrophage nucleus >=75% contained by source cell polygon",
        "cases": len(summary), "records": summary}, indent=2) + "\n")
    counts = {name: sum(row["matched_by_type"][name] for row in summary) for name in CLASSES}
    print(json.dumps({"cases": len(summary), "matched_by_type": counts,
                      "ambiguous_macrophage_regions": sum(row["ambiguous_macrophage_regions"]
                                                        for row in summary)}))


if __name__ == "__main__":
    main()
