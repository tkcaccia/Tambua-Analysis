#!/usr/bin/env python3
"""Render all corrected-QC StarDist cell predictions from a fitted fastPLS model.

Only saved 23-channel GigaTIME scores are read. Fit-image predictions are
in-sample; development and organizer-test images are patient-separated from
the fit. The gallery never presents unmatched detections as verified errors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Gallery generation must explicitly hide all GPUs")

import numpy as np
from PIL import Image
import tifffile


COLORS = {"epithelial": (251, 149, 72), "lymphocyte": (66, 193, 220),
          "macrophage": (188, 126, 224), "neutrophil": (136, 207, 108)}
ERROR_COLORS = {"correct": (68, 196, 116), "wrong": (245, 90, 79)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def overlay(image: np.ndarray, labels: np.ndarray,
            values: dict[int, str], palette: dict[str, tuple[int, int, int]],
            *, alpha: float = .38) -> Image.Image:
    if image.shape[:2] != labels.shape or image.dtype != np.uint8:
        raise ValueError("H&E and StarDist masks are not aligned RGB pixels")
    result = image.copy()
    colors = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    present = np.zeros(len(colors), dtype=bool)
    for label_id, name in values.items():
        if label_id < 1 or label_id >= len(colors) or name not in palette:
            raise ValueError("Invalid predicted cell ID or type")
        colors[label_id] = palette[name]
        present[label_id] = True
    include = present[labels]
    if np.any(include):
        result[include] = np.round((1-alpha) * image[include] +
                                   alpha * colors[labels[include]]).astype(np.uint8)
        boundary = include & ((labels != np.roll(labels, 1, 0)) |
                              (labels != np.roll(labels, -1, 0)) |
                              (labels != np.roll(labels, 1, 1)) |
                              (labels != np.roll(labels, -1, 1)))
        result[boundary] = colors[labels[boundary]]
    return Image.fromarray(result, mode="RGB")


def detected_mask_image(labels: np.ndarray) -> Image.Image:
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected a 2-D StarDist instance mask")
    result = np.full((*labels.shape, 3), 248, dtype=np.uint8)
    positive = labels > 0
    result[positive] = (43, 97, 124)
    boundary = positive & ((labels != np.roll(labels, 1, 0)) |
                           (labels != np.roll(labels, -1, 0)) |
                           (labels != np.roll(labels, 1, 1)) |
                           (labels != np.roll(labels, -1, 1)))
    result[boundary] = (255, 255, 255)
    return Image.fromarray(result, mode="RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-summary", type=Path, required=True)
    ap.add_argument("--quant-root", type=Path, required=True)
    ap.add_argument("--score-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--tambua-src", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    study = json.loads((args.score_root / "DETECTED_CELL_STUDY_COMPLETE.json").read_text())
    qc = json.loads(args.qc_summary.read_text())
    selected = [row for row in qc["records"]
                if all(value == "pass" for value in row["stardist_nuclear_qc"].values())]
    if study["status"] != "complete" or len(selected) != 47 or \
            Counter(row["split"] for row in selected) != \
            {"fit": 33, "development": 4, "test": 10}:
        raise ValueError("The corrected detected-cell study has changed")
    args.output.mkdir(parents=True)
    manifest = args.output / "inference_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "patient_id",
                                                   "nuclei_csv", "cell_csv"))
        writer.writeheader()
        for row in selected:
            sample = row["sample_id"]
            root = args.quant_root / sample
            if not (root / "CASE_COMPLETE.json").is_file():
                raise FileNotFoundError(f"Missing quantified detected-cell case: {sample}")
            writer.writerow({"sample_id": sample, "patient_id": row["patient_id"],
                             "nuclei_csv": root / "detected_nuclei_mean_intensity.csv",
                             "cell_csv": root / "detected_cell_mean_intensity.csv"})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.tambua_src.resolve())
    table = args.output / "all_qc_detected_cells.csv"
    predictions_path = args.output / "all_qc_predictions.csv"
    def cli(*values: str) -> None:
        subprocess.run([sys.executable, "-m", "tambua.cli", *values],
                       check=True, env=env, stdout=subprocess.DEVNULL)
    cli("build-inference", "--manifest", str(manifest), "--output", str(table),
        "--compartments", "nuclei-cell")
    cli("predict", "--table", str(table), "--model",
        str(args.score_root / "detected_cell_classifier.rds"),
        "--output", str(predictions_path))
    by_sample: dict[str, dict[int, str]] = defaultdict(dict)
    for row in read_csv(predictions_path):
        sample, label_id = row["sample_id"], int(row["label_id"])
        if label_id in by_sample[sample]:
            raise ValueError("Duplicate detected-cell prediction")
        by_sample[sample][label_id] = row["predicted_cell_type"]
    expected = {(row["sample_id"], row["label_id"]): row["predicted_cell_type"]
                for row in read_csv(args.score_root / "all_detected_test_predictions.csv")}
    actual = {(sample, str(label_id)): name for sample, ids in by_sample.items()
              for label_id, name in ids.items() if
              any(row["sample_id"] == sample and row["split"] == "test" for row in selected)}
    if actual != expected:
        raise ValueError("Gallery test predictions differ from the frozen test evaluation")
    records = []
    for row in selected:
        sample = row["sample_id"]
        source = args.source_root / "prepared" / sample
        quantified = args.quant_root / sample
        image = tifffile.imread(source / "he.ome.tif")
        detected = tifffile.imread(quantified / "stardist_nuclei.ome.tif")
        expert = {int(item["label_id"]): item["cell_type"] for item in
                  read_csv(quantified / "expert_types_on_detected.csv")}
        predicted = by_sample[sample]
        if len(predicted) != len(np.unique(detected)) - 1 or \
                set(predicted) != set(int(v) for v in np.unique(detected) if v > 0):
            raise ValueError(f"Missing prediction on a detected nucleus: {sample}")
        verdict = {label_id: "correct" if predicted[label_id] == name else "wrong"
                   for label_id, name in expert.items()}
        overlay(image, detected, predicted, COLORS).save(
            args.output / f"{sample}-predicted.webp", format="WEBP", quality=85)
        detected_mask_image(detected).save(
            args.output / f"{sample}-detected.webp", format="WEBP", quality=85)
        overlay(image, detected, verdict, ERROR_COLORS, alpha=.50).save(
            args.output / f"{sample}-matched-errors.webp", format="WEBP", quality=85)
        correct = sum(name == predicted[label_id] for label_id, name in expert.items())
        support = Counter(expert.values())
        recall = {name: (sum(expert.get(label_id) == name and predicted[label_id] == name
                             for label_id in expert) / support[name] if support[name] else None)
                  for name in COLORS}
        records.append({"sample_id": sample, "patient_id": row["patient_id"],
                        "organ": row["organ"], "split": row["split"],
                        "detected_cells_predicted": len(predicted),
                        "conservatively_matched_expert_cells": len(expert),
                        "matched_accuracy": correct / len(expert) if expert else None,
                        "expert_support": {name: support[name] for name in COLORS},
                        "expert_recall": recall,
                        "source_mpp": json.loads((source / "provenance.json").read_text())["mpp"],
                        "width": int(image.shape[1]), "height": int(image.shape[0]),
                        "star_dist_reference_qc": row["stardist_nuclear_qc"]})
    report = {"status": "complete", "device": "CPU", "cases": len(records),
              "cohort": "corrected scanner RGB, 0.25 µm/pixel GigaTIME, StarDist-reference QC",
              "prediction_status": {"fit": "in-sample fitted predictions",
                                    "development": "patient-separated exploratory predictions",
                                    "test": "patient-separated organizer-test predictions"},
              "all_detected_test_cells_predicted": study["all_detected_test_cells_predicted"],
              "test_metrics_on_matched_cells": study["test_metrics_on_conservatively_matched_cells"],
              "limit": "Unmatched detections have predictions but no expert type; their accuracy is unknown. No matched test neutrophils, so four-class balanced accuracy is undefined.",
              "records": records}
    staged = args.output / "manifest.json.tmp"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(args.output / "manifest.json")
    print(json.dumps({"cases": len(records),
                      "by_split": dict(Counter(row["split"] for row in records)),
                      "test_cells": study["all_detected_test_cells_predicted"]}, indent=2))


if __name__ == "__main__":
    main()
