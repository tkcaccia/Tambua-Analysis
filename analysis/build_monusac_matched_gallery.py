#!/usr/bin/env python3
"""Build all 32 expert-annotated held-out MoNuSAC image panels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from build_monusac_gallery import CLASSES, COLORS, dapi_qc, save_webp
from report_class_coverage import report as class_coverage_report


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def overlay(he: Image.Image, ids: np.ndarray, labels: dict[int, str],
            *, actual: dict[int, str] | None = None) -> Image.Image:
    rgb = np.asarray(he.convert("RGB"), dtype=np.uint8)
    if ids.shape != rgb.shape[:2]:
        raise ValueError("Image and expert nuclear mask are not registered")
    palette = np.zeros((int(ids.max()) + 1, 3), dtype=np.uint8)
    if actual is None:
        palette[1:] = (120, 120, 120)  # gray: no matched quantification
        for index, name in labels.items():
            palette[index] = COLORS[name]
    else:
        palette[1:] = (120, 120, 120)
        for index, name in actual.items():
            if index in labels:
                palette[index] = (65, 186, 111) if labels[index] == name else (226, 70, 81)
    output = rgb.copy()
    mask = ids > 0
    output[mask] = np.rint(rgb[mask] * 0.25 + palette[ids][mask] * 0.75).astype(np.uint8)
    return Image.fromarray(output)


def build(source: Path, scores: Path, site: Path) -> dict:
    cohort = json.loads((source / "input/COHORT.json").read_text())
    report = json.loads((scores / "report.json").read_text())
    if not (source / "evidence/BUNDLE_COMPLETE.json").is_file():
        raise ValueError("Inference bundle incomplete")
    expert_rows = csv_rows(scores / "cell_table.csv")
    prediction_rows = csv_rows(scores / "predictions.csv")
    expert = {(row["sample_id"], int(row["label_id"])): row["cell_type"]
              for row in expert_rows}
    predicted = {(row["sample_id"], int(row["label_id"])): row["predicted_cell_type"]
                 for row in prediction_rows}
    if len(expert) != len(expert_rows) or len(predicted) != len(prediction_rows) or set(expert) != set(predicted):
        raise ValueError("Matched labels and predictions do not align one-to-one")
    output = site / "assets/monusac-matched"
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in cohort["records"]:
        if row["split"] != "test":
            continue
        sample = row["sample_id"]
        prepared = source / "prepared" / sample
        generated = source / "output" / sample
        if not (generated / "RUN_COMPLETE.json").is_file():
            raise ValueError(f"Inference incomplete: {sample}")
        metadata = json.loads((generated / "gigatime_output/gigatime_metadata.json").read_text())
        if abs(metadata["source_mpp"] - row["mpp"]) > 1e-6 or \
                abs(metadata["effective_mpp"] - 0.25) > 1e-6:
            raise ValueError(f"Pixel spacing not calibrated: {sample}")
        with Image.open(prepared / "he.ome.tif") as image:
            he = image.convert("RGB")
        with Image.open(prepared / "expert_nuclei.ome.tif") as image:
            ids = np.asarray(image, dtype=np.int32)
        with Image.open(prepared / "expert_anchored_cell_proxy.ome.tif") as image:
            cell_ids = np.asarray(image, dtype=np.int32)
        reference = {int(r["label_id"]): r["cell_type"] for r in csv_rows(prepared / "expert_types.csv")}
        matched = {index: name for (case, index), name in expert.items() if case == sample}
        predictions = {index: name for (case, index), name in predicted.items() if case == sample}
        if set(np.unique(ids)) - {0} != set(reference) or not matched or \
                set(matched) != set(predictions) or not set(matched) <= set(reference):
            raise ValueError(f"Expert geometry and classified cells disagree: {sample}")
        panels = {"he": he, "expert": overlay(he, ids, reference),
                  "predicted": overlay(he, ids, predictions),
                  "errors": overlay(he, ids, predictions, actual=reference)}
        marker_root = generated / "gigatime_output/jpg_image"
        with Image.open(marker_root / "DAPI.jpg") as image:
            qc = dapi_qc(ids, cell_ids, image.convert("L"))
        for marker in ("DAPI", "CK", "CD3", "CD68"):
            with Image.open(marker_root / f"{marker}.jpg") as image:
                panels[marker.lower()] = image.convert("L")
        for name, image in panels.items():
            save_webp(image, output / f"{sample}-{name}.webp")
        confusion = [[sum(matched[index] == actual and predictions[index] == prediction
                          for index in matched) for prediction in CLASSES]
                     for actual in CLASSES]
        support = Counter(matched.values())
        correct = sum(matched[index] == predictions[index] for index in matched)
        cases.append({"sample_id": sample, "patient_id": row["patient_id"],
                      "organ": row["organ"], "source_mpp": row["mpp"], "model_mpp": 0.25,
                      "width": row["width"], "height": row["height"],
                      "annotated_cells": len(reference), "n_cells": len(matched),
                      "correct": correct, "accuracy": round(correct / len(matched), 5),
                      "support": {name: support[name] for name in CLASSES},
                      "confusion_matrix": confusion, "dapi_qc": qc})
    if len(cases) != 32 or sum(item["n_cells"] for item in cases) != report["test_cells"]:
        raise ValueError("Gallery does not cover all 32 test images and scored cells")
    if not np.array_equal(np.sum([case["confusion_matrix"] for case in cases], axis=0),
                          np.asarray(report["test"]["confusion_matrix"])):
        raise ValueError("Gallery confusion disagrees with score report")
    result = {"cohort": "MoNuSAC four-tissue patient-disjoint test",
              "train_images": 88, "test_images": 32, "train_patients": 40,
              "test_patients": 19, "train_cells": report["train_cells"],
              "test_cells": report["test_cells"], "test": report["test"],
              "test_by_tissue": report["test_by_tissue"],
              "test_equal_patient_accuracy": report["test_by_patient"]["equal_weight_accuracy"],
              "majority_baseline": report["majority_baseline"],
              "dapi_qc_passed_test_images": sum(x["dapi_qc"]["passes_enrichment_screen"] for x in cases),
              "class_order": CLASSES,
              "class_coverage": class_coverage_report(cases, CLASSES),
              "scope": "Conditional cell typing at expert nuclei, with a 3 µm nucleus-anchored cell proxy. Virtual markers are uncalibrated predictions.",
              "cases": cases}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.results, args.scores, args.site)
    print(json.dumps({"test_images": result["test_images"],
                      "test_cells": result["test_cells"],
                      "accuracy": result["test"]["accuracy"]}))
