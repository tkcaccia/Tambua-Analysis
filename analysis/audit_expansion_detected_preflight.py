#!/usr/bin/env python3
"""Audit expert-to-StarDist matching in the prepared 93-image expansion.

This is a CPU-only preflight. It does not infer GigaTIME markers or admit any
image through nuclear/shadow QC; all counts remain provisional.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")
SPLITS = ("train", "development", "test")
ORGANS = ("breast", "kidney", "lung", "prostate")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(prepared: Path, fit: Path, evaluation: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Expansion preflight must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    marker = prepared / "PREP_COMPLETE.json"
    cohort = json.loads(marker.read_text())
    if cohort["cases"] != 93 or not cohort["all_scanner_rgb"]:
        raise ValueError("Expanded source cohort changed")
    selected = {row["sample_id"]: row for row in cohort["records"]}
    if len(selected) != 93:
        raise ValueError("Duplicate expansion sample ID")
    paths = (fit / "SUMMARY.json", evaluation / "SUMMARY.json")
    rows = [row for path in paths for row in json.loads(path.read_text())["records"]
            if row["sample_id"] in selected]
    if len(rows) != 93 or {row["sample_id"] for row in rows} != set(selected):
        raise ValueError("StarDist matching does not cover every expanded image")
    per_image = []
    for row in rows:
        sample = row["sample_id"]
        source = selected[sample]
        split = row["split"]
        if split not in SPLITS or row["organ"] != source["organ"] or \
                row["patient_id"] != source["patient_id"] or \
                source["source_color_reference"] != "scanner_svs_rgb" or \
                not .15 <= float(source["mpp"]) <= .4:
            raise ValueError(f"Split or source identity mismatch: {sample}")
        folder = (fit if split == "train" else evaluation)
        detail = json.loads((folder / f"{sample}.json").read_text())
        assignments = detail["assignments"]
        if Counter(item["cell_type"] for item in assignments) != Counter(
                {name: row["matched_by_type"][name] for name in CLASSES}):
            raise ValueError(f"Assignment and summary counts differ: {sample}")
        mask = tifffile.imread(prepared / "samples" / sample / "expert_nuclei.ome.tif")
        areas = np.bincount(mask.ravel())
        flagged = int(sum(item["cell_type"] == "macrophage" and
                          areas[item["expert_id"]] > 5000 for item in assignments))
        per_image.append({"sample_id": sample, "patient_id": row["patient_id"],
                          "organ": row["organ"], "split": split,
                          "matched_by_type": row["matched_by_type"],
                          "source_expert_by_type": row["expert_by_type"],
                          "matched_macrophage_polygons_gt_5000_px": flagged,
                          "ambiguous_macrophage_regions_withheld":
                              row["ambiguous_macrophage_regions"]})
    counts = {}
    for split in SPLITS:
        subset = [row for row in per_image if row["split"] == split]
        counts[split] = {
            "images": len(subset), "patients": len({row["patient_id"] for row in subset}),
            "source_expert_by_type": {name: sum(row["source_expert_by_type"][name]
                                               for row in subset) for name in CLASSES},
            "matched_by_type": {name: sum(row["matched_by_type"][name] for row in subset)
                                for name in CLASSES},
            "matched_macrophage_polygons_gt_5000_px": sum(
                row["matched_macrophage_polygons_gt_5000_px"] for row in subset),
            "ambiguous_macrophage_regions_withheld": sum(
                row["ambiguous_macrophage_regions_withheld"] for row in subset),
            "images_with_matched_rare_class": {name: sum(
                row["matched_by_type"][name] > 0 for row in subset)
                for name in ("macrophage", "neutrophil")},
            "by_organ": {organ: {"images": sum(row["organ"] == organ for row in subset),
                                 "macrophages": sum(row["matched_by_type"]["macrophage"]
                                                    for row in subset if row["organ"] == organ),
                                 "neutrophils": sum(row["matched_by_type"]["neutrophil"]
                                                    for row in subset if row["organ"] == organ)}
                         for organ in ORGANS},
        }
    if [counts[split]["images"] for split in SPLITS] != [44, 15, 34]:
        raise ValueError("Frozen fit/development/test partition changed")
    patient_sets = {split: {row["patient_id"] for row in per_image if row["split"] == split}
                    for split in SPLITS}
    if any(patient_sets[left] & patient_sets[right] for left, right in
           (("train", "development"), ("train", "test"), ("development", "test"))):
        raise ValueError("Patient leakage across expansion splits")
    report = {"status": "complete", "device": "CPU", "images": 93,
              "patient_disjoint_splits": True,
              "source_prep_sha256": sha(marker),
              "matching_summary_sha256": {"fit": sha(paths[0]), "evaluation": sha(paths[1])},
              "counts": counts, "records": sorted(per_image, key=lambda row: row["sample_id"]),
              "rule": "One-to-one nuclear IoU>=0.5 or exactly one unassigned macrophage nucleus >=75% inside its expert cell polygon",
              "oversized_rule": "Matched macrophage expert polygons over 5000 native pixels are flagged for manual review, not counted as validated nuclear shapes",
              "limit": "Before GigaTIME inference and nuclear/shadow QC. Counts are candidate matched labels, not retained fit/test cells, classifier accuracy or proof of independent generalization."}
    output.mkdir(parents=True)
    (output / "EXPANSION_DETECTED_PREFLIGHT.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.prepared, args.fit, args.evaluation, args.output)
    print(json.dumps({split: result["counts"][split] for split in SPLITS}, indent=2))
