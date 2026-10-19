#!/usr/bin/env python3
"""Describe expert cell-type coverage before interpreting classification scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def score_case(case: dict, classes: list[str], minimum_support: int = 5) -> dict:
    matrix = np.asarray(case["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or np.any(matrix < 0):
        raise ValueError("Invalid per-image confusion matrix")
    support = matrix.sum(axis=1)
    if support.sum() != case["n_cells"]:
        raise ValueError(f"Confusion matrix and cell count differ: {case['sample_id']}")
    if any(support[i] != case["support"][name] for i, name in enumerate(classes)):
        raise ValueError(f"Confusion matrix and class support differ: {case['sample_id']}")
    present = np.flatnonzero(support > 0)
    sufficient = np.flatnonzero(support >= minimum_support)

    def mean_recall(indices: np.ndarray) -> float | None:
        if len(indices) < 2:
            return None
        return float(np.mean([matrix[i, i] / support[i] for i in indices]))

    def macro_f1(indices: np.ndarray) -> float | None:
        if len(indices) < 2:
            return None
        column = matrix.sum(axis=0)
        return float(np.mean([2 * matrix[i, i] / (support[i] + column[i])
                              if support[i] + column[i] else 0.0 for i in indices]))

    return {
        "sample_id": case["sample_id"],
        "organ": case.get("organ"),
        "n_cells": int(support.sum()),
        "reference_class_count": len(present),
        "reference_classes": [classes[i] for i in present],
        "classes_with_at_least_five_cells": [classes[i] for i in sufficient],
        "multiclass_eligible": len(present) >= 2,
        "multiclass_minimum_five_eligible": len(sufficient) >= 2,
        "observed_class_macro_recall": mean_recall(present),
        "observed_class_macro_f1": macro_f1(present),
        "minimum_five_class_macro_recall": mean_recall(sufficient),
        "minimum_five_class_macro_f1": macro_f1(sufficient),
        "accuracy": float(np.trace(matrix) / support.sum()),
    }


def report(cases: list[dict], classes: list[str], minimum_support: int = 5) -> dict:
    records = [score_case(case, classes, minimum_support) for case in cases]
    if not records:
        raise ValueError("Empty test cohort")
    distribution = Counter(row["reference_class_count"] for row in records)

    def subset(key: str) -> dict:
        selected = [row for row in records if row[key]]
        return {
            "images": len(selected),
            "cells": sum(row["n_cells"] for row in selected),
            "equal_image_observed_class_macro_recall": (
                float(np.mean([row["observed_class_macro_recall"] for row in selected]))
                if selected else None
            ),
            "equal_image_observed_class_macro_f1": (
                float(np.mean([row["observed_class_macro_f1"] for row in selected]))
                if selected else None
            ),
            "equal_image_minimum_five_class_macro_recall": (
                float(np.mean([row["minimum_five_class_macro_recall"] for row in selected]))
                if selected and key == "multiclass_minimum_five_eligible" else None
            ),
            "equal_image_minimum_five_class_macro_f1": (
                float(np.mean([row["minimum_five_class_macro_f1"] for row in selected]))
                if selected and key == "multiclass_minimum_five_eligible" else None
            ),
        }

    single = [row for row in records if row["reference_class_count"] == 1]
    return {
        "schema_version": 1,
        "minimum_per_class_support_sensitivity": minimum_support,
        "images": len(records),
        "cells": sum(row["n_cells"] for row in records),
        "reference_class_count_distribution": {str(key): distribution[key] for key in sorted(distribution)},
        "single_class_images": len(single),
        "single_class_cells": sum(row["n_cells"] for row in single),
        "multiclass": subset("multiclass_eligible"),
        "multiclass_minimum_five": subset("multiclass_minimum_five_eligible"),
        "per_image": records,
        "interpretation": (
            "Only images with at least two expert reference classes contribute to the "
            "multiclass image-average scores. Classes with fewer than five expert "
            "cells are excluded from the sensitivity score. Single-class images "
            "remain visible but have no multiclass score. The score is conditional "
            "on expert nuclear masks, and does not assess cell detection."
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(args.results.read_text())
    result = report(source["cases"], source["class_order"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("images", "single_class_images", "multiclass", "multiclass_minimum_five")}, indent=2))
