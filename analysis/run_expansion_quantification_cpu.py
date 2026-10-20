#!/usr/bin/env python3
"""Quantify all QC-passing expansion cases over independent StarDist cells.

Requires the complete CPU-only GigaTIME bundle. Failed DAPI/shadow images are
never quantified for classifier fitting or test scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path

from quantify_detected_monusac_cells import (
    run as quantify_one, sha256, validate_quantified_table,
)


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Expansion quantification must hide GPUs")
    bundle = json.loads((args.inference / "BUNDLE_COMPLETE.json").read_text())
    if bundle["status"] != "complete" or bundle["device"] != "CPU" or \
            bundle["cases"] != 93 or len(bundle["case_order"]) != 93:
        raise ValueError("Complete 93-image CPU inference is required")
    prep = json.loads((args.prepared / "PREP_COMPLETE.json").read_text())
    source = {row["sample_id"]: row for row in prep["records"]}
    matches = {row["sample_id"]: (row, root) for root in
               (args.fit_match, args.eval_match) for row in
               json.loads((root / "SUMMARY.json").read_text())["records"]
               if row["sample_id"] in source}
    if len(source) != 93 or len(matches) != 93 or set(bundle["case_order"]) != set(source):
        raise ValueError("Prepared, matched and inferred image identities differ")
    if (args.output / "COHORT_COMPLETE.json").exists():
        raise FileExistsError("Expansion quantification already complete")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(bundle["case_order"], 1):
        case = args.inference / sample
        checkpoint = json.loads((case / "CASE_COMPLETE.json").read_text())
        if checkpoint["sample_id"] != sample:
            raise ValueError("Wrong case checkpoint identity")
        qc = checkpoint["nuclear_qc"]
        if any(qc[field] != "pass" for field in
               ("status", "alignment_status", "border_shadow_status",
                "spatial_shadow_status")):
            print(f"EXCLUDED {index}/93 {sample} QC={qc['status']}", flush=True)
            continue
        matched, match_root = matches[sample]
        detected_root = args.fit_detected if matched["split"] == "train" else args.eval_detected
        detected = detected_root / sample / "stardist_labels.tif"
        output = args.output / sample
        if output.exists():
            marker = output / "CASE_COMPLETE.json"
            if not marker.is_file():
                raise ValueError(f"Incomplete existing quantification: {sample}")
            result = json.loads(marker.read_text())
            if result["sample_id"] != sample or result["status"] != "complete" or \
                    result.get("quantifier_sha256") != sha256(args.quantifier):
                raise ValueError("Invalid reused quantification checkpoint")
        else:
            result = quantify_one(sample, detected, match_root / f"{sample}.json",
                                  args.prepared / "samples" / sample,
                                  case, args.quantifier, output,
                                  qc_report_path=case / "stardist_qc.json")
        for compartment in ("nuclei", "cell"):
            validate_quantified_table(
                output / f"detected_{compartment}_mean_intensity.csv",
                output / f"detected_{compartment}_summary.json", compartment)
        with (output / "expert_types_on_detected.csv").open(newline="") as stream:
            labels = list(csv.DictReader(stream))
        counts = Counter(row["cell_type"] for row in labels)
        if sum(counts.values()) != result["expert_types_conservatively_matched"]:
            raise ValueError("Matched label count differs from quantified case")
        if any(counts[name] != matched["matched_by_type"][name] or
               counts[name] > matched["expert_by_type"][name] for name in CLASSES):
            raise ValueError("Matched-cell or expert support differs from frozen nuclear matching")
        flagged = sum(row["cell_type"] == "macrophage" and
                      int(row["expert_area_native_px"]) > 5000 for row in labels)
        rows.append({"sample_id": sample, "patient_id": matched["patient_id"],
                     "split": matched["split"], "organ": matched["organ"],
                     "detected_cells": result["detected_cells_quantified"],
                     "expert_by_type": {name: matched["expert_by_type"][name]
                                        for name in CLASSES},
                     "matched_by_type": {name: counts[name] for name in CLASSES},
                     "unmatched_detected_instances": matched["unmatched_detected_instances"],
                     "large_macrophage_regions_flagged": flagged})
        print(f"QUANTIFIED {index}/93 {sample} matched={len(labels)}", flush=True)
    split_patients = {split: {row["patient_id"] for row in rows if row["split"] == split}
                      for split in ("train", "development", "test")}
    if any(split_patients[a] & split_patients[b] for a, b in
           (("train", "development"), ("train", "test"), ("development", "test"))):
        raise ValueError("Patient leakage among QC-passing expansion cases")
    report = {"status": "complete", "device": "CPU",
              "quantifier_sha256": sha256(args.quantifier),
              "source_images": 93, "qc_passing_images": len(rows),
              "by_split": dict(Counter(row["split"] for row in rows)),
              "matched_by_type": {split: {name: sum(row["matched_by_type"][name]
                                                         for row in rows if row["split"] == split)
                                         for name in CLASSES}
                                  for split in ("train", "development", "test")},
              "expert_by_type": {split: {name: sum(row["expert_by_type"][name]
                                                        for row in rows if row["split"] == split)
                                        for name in CLASSES}
                                 for split in ("train", "development", "test")},
              "large_macrophage_regions_flagged": sum(
                  row["large_macrophage_regions_flagged"] for row in rows),
              "records": rows,
              "limit": "No fastPLS score yet. Matched labels remain conditional on StarDist detection and expert-region matching; oversized macrophage labels require sensitivity analysis."}
    staged = args.output / "COHORT_COMPLETE.json.tmp"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(args.output / "COHORT_COMPLETE.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "inference", "fit-match", "eval-match",
                 "fit-detected", "eval-detected", "quantifier", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in
                      ("qc_passing_images", "by_split", "matched_by_type")}, indent=2))
