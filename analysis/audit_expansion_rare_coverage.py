#!/usr/bin/env python3
"""Audit expert class coverage of the prepared 93-image MoNuSAC expansion.

Counts refer to source polygons before StarDist matching, GigaTIME inference or
quality control. They establish cohort value, not a classifier outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def audit(candidate: Path, current_train: Path, current_test: Path) -> dict:
    proposed = json.loads(candidate.read_text())
    if proposed.get("additional_samples") != 93 or len(proposed["records"]) != 213:
        raise ValueError("Frozen 213-image candidate changed")
    current = {row["sample_id"] for source in (current_train, current_test)
               for row in json.loads(source.read_text())["cases"]}
    if len(current) != 120:
        raise ValueError("Frozen 120-image cohort changed")
    rows = proposed["records"]
    if len({row["sample_id"] for row in rows}) != 213 or \
            not current <= {row["sample_id"] for row in rows}:
        raise ValueError("Candidate missing or duplicating source images")
    patients = {split: {row["patient_id"] for row in rows if row["split"] == split}
                for split in ("train", "test")}
    if patients["train"] & patients["test"]:
        raise ValueError("Training and test patients overlap")
    organs = {split: {row["organ"] for row in rows if row["split"] == split}
              for split in ("train", "test")}
    if organs["train"] != organs["test"]:
        raise ValueError("Training and test tissues differ")
    table = {}
    for split in ("train", "test"):
        for scope in ("current", "additional", "expanded"):
            chosen = [row for row in rows if row["split"] == split and
                      (scope == "expanded" or (row["sample_id"] in current) ==
                       (scope == "current"))]
            counts = Counter()
            for row in chosen:
                counts.update(row["expert_types"])
            table[f"{split}_{scope}"] = {
                "images": len(chosen),
                "patients": len({row["patient_id"] for row in chosen}),
                "source_polygons_by_class": {name: counts[name] for name in CLASSES},
                "images_with_neutrophils": sum(row["expert_types"].get("neutrophil", 0) > 0
                                               for row in chosen),
                "images_with_macrophages": sum(row["expert_types"].get("macrophage", 0) > 0
                                               for row in chosen),
            }
    if table["train_additional"]["images"] != 59 or \
            table["test_additional"]["images"] != 34:
        raise ValueError("Additional fit/test size changed")
    return {"status": "complete", "scope": "CPU audit of source annotations only",
            "candidate_images": 213, "current_images": 120,
            "additional_images": 93,
            "counts": table,
            "rare_additional_test_cases": [
                {"sample_id": row["sample_id"], "patient_id": row["patient_id"],
                 "organ": row["organ"], "macrophages": row["expert_types"].get("macrophage", 0),
                 "neutrophils": row["expert_types"].get("neutrophil", 0)}
                for row in rows if row["split"] == "test" and row["sample_id"] not in current
                and (row["expert_types"].get("macrophage", 0) or
                     row["expert_types"].get("neutrophil", 0))],
            "limit": "Source polygon counts precede independent nuclear detection, one-to-one matching, GigaTIME inference and DAPI/shadow QC. Rare-class evaluability may be much lower."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--current-train", type=Path, required=True)
    ap.add_argument("--current-test", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.candidate, args.current_train, args.current_test)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("additional_images", "counts")}))
