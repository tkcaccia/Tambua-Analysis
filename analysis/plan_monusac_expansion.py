#!/usr/bin/env python3
"""Freeze a patient-disjoint, calibrated expansion candidate without inference.

The source audits come from the original organizer archives. Existing 120
cases must remain a subset, and training patients present anywhere in the
original test archive are excluded before the size gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(train_path: Path, test_path: Path, current_path: Path,
         min_side: int = 256) -> dict:
    if min_side < 256:
        raise ValueError("A 256-native-pixel GigaTIME input is the minimum candidate")
    train = json.loads(train_path.read_text())["records"]
    test = json.loads(test_path.read_text())["records"]
    current = json.loads(current_path.read_text())["records"]
    test_patients = {r["patient_id"] for r in test}
    rows = ([dict(r, split="train") for r in train
             if min(r["width"], r["height"]) >= min_side and
             r["patient_id"] not in test_patients] +
            [dict(r, split="test") for r in test
             if min(r["width"], r["height"]) >= min_side])
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    old = {r["sample_id"]: r["split"] for r in current}
    if not set(old) <= set(ids) or any(r["split"] != old.get(r["sample_id"], r["split"])
                                   for r in rows):
        raise ValueError("Candidate must preserve every existing case and split")
    splits = {split: [r for r in rows if r["split"] == split]
              for split in ("train", "test")}
    if {r["patient_id"] for r in splits["train"]} & \
            {r["patient_id"] for r in splits["test"]}:
        raise ValueError("Patient leakage")
    if {r["organ"] for r in splits["test"]} - {r["organ"] for r in splits["train"]}:
        raise ValueError("A test tissue is absent from training")
    for organ in {r["organ"] for r in splits["test"]}:
        train_types = {name for r in splits["train"] if r["organ"] == organ
                       for name in CLASSES if r["expert_types"].get(name, 0)}
        test_types = {name for r in splits["test"] if r["organ"] == organ
                      for name in CLASSES if r["expert_types"].get(name, 0)}
        if not test_types <= train_types:
            raise ValueError(f"Test-only expert class in {organ}")
    summary = {}
    for split, items in splits.items():
        summary[split] = {
            "images": len(items), "patients": len({r["patient_id"] for r in items}),
            "by_tissue": dict(sorted(Counter(r["organ"] for r in items).items())),
            "expert_regions_by_class": {name: sum(r["expert_types"][name] for r in items)
                                        for name in CLASSES},
        }
    return {
        "status": "prospective_candidate_no_gigatime_inference_or_quality_score",
        "min_native_side_px": min_side,
        "source_sha256": {"train_audit": digest(train_path), "test_audit": digest(test_path),
                          "current_120_cohort": digest(current_path)},
        "current_samples_preserved": len(old),
        "additional_samples": len(ids) - len(old),
        "summary": summary,
        "records": [{key: r[key] for key in ("sample_id", "patient_id", "organ", "split",
                                              "width", "height", "mpp", "expert_types")}
                    for r in rows],
        "interpretation": "Training and test patients are disjoint and all test tissues/classes occur in training. New ROI inference, RGB verification, measured-resolution check, visual review and native DAPI/shadow QC are required before scores may be reported.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--test-audit", type=Path, required=True)
    parser.add_argument("--current-cohort", type=Path, required=True)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.train_audit, args.test_audit, args.current_cohort,
                  args.min_side)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"additional_samples": result["additional_samples"],
                      "summary": result["summary"]}, indent=2))
