#!/usr/bin/env python3
"""Check whether corrected virtual-score metadata changed any saved cell value."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from quantify_detected_monusac_cells import GIGATIME_CHANNELS


def table(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = {int(row["label_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate detected-cell IDs: {path}")
    return result


def compare(old: Path, new: Path) -> dict:
    original = json.loads((old / "COHORT_COMPLETE.json").read_text())
    corrected = json.loads((new / "COHORT_COMPLETE.json").read_text())
    old_ids = {row["sample_id"] for row in original["records"]}
    new_ids = {row["sample_id"] for row in corrected["records"]}
    if original["qc_passing_images"] != 9 or old_ids != new_ids or \
            corrected.get("quantifier_sha256") is None:
        raise ValueError("Old and new nine-field cohorts differ")
    maximum, total, count = 0.0, 0.0, 0
    per_case = []
    for sample in sorted(old_ids):
        case_maximum = 0.0
        cell_count = None
        for compartment in ("nuclei", "cell"):
            name = f"detected_{compartment}_mean_intensity.csv"
            previous = table(old / sample / name)
            current = table(new / sample / name)
            if previous.keys() != current.keys() or not previous:
                raise ValueError(f"Detected-cell IDs changed: {sample} {compartment}")
            cell_count = len(current)
            for index in current:
                for channel in GIGATIME_CHANNELS:
                    difference = abs(float(previous[index][channel]) -
                                     float(current[index][channel]))
                    maximum = max(maximum, difference)
                    case_maximum = max(case_maximum, difference)
                    total += difference
                    count += 1
        per_case.append({"sample_id": sample, "detected_cells": cell_count,
                         "maximum_absolute_score_change": case_maximum})
    return {"status": "complete", "cases": 9,
            "values_compared": count,
            "maximum_absolute_score_change": maximum,
            "mean_absolute_score_change": total / count,
            "same_numeric_scores": maximum == 0.0,
            "new_quantifier_sha256": corrected["quantifier_sha256"],
            "records": per_case}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("cases", "values_compared", "maximum_absolute_score_change")}))
