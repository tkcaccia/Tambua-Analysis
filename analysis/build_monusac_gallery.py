#!/usr/bin/env python3
"""Build a static, auditable MoNuSAC cross-organ cell-type result gallery.

Only derived, downsampled image panels are published. The original data stay
outside Git; their terms remain with the MoNuSAC organizers.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")
COLORS = {
    "epithelial": (251, 149, 72),
    "lymphocyte": (66, 193, 220),
    "macrophage": (188, 126, 224),
    "neutrophil": (136, 207, 108),
}
MAX_SIDE = 1100


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save_webp(image: Image.Image, path: Path) -> None:
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    image.save(path, "WEBP", quality=80, method=6)


def color_overlay(he: Image.Image, ids: np.ndarray, labels: dict[int, str]) -> Image.Image:
    rgb = np.asarray(he.convert("RGB"), dtype=np.uint8)
    if ids.shape != rgb.shape[:2]:
        raise ValueError("Image and expert nuclear mask are not registered")
    if set(np.unique(ids)) - {0} != set(labels):
        raise ValueError("Every expert nucleus must have exactly one cell label")
    palette = np.zeros((int(ids.max()) + 1, 3), dtype=np.uint8)
    for index, name in labels.items():
        palette[index] = COLORS[name]
    colored = palette[ids]
    output = rgb.copy()
    mask = ids > 0
    output[mask] = np.rint(rgb[mask] * 0.25 + colored[mask] * 0.75).astype(np.uint8)
    return Image.fromarray(output)


def error_overlay(he: Image.Image, ids: np.ndarray, actual: dict[int, str],
                  predicted: dict[int, str]) -> Image.Image:
    rgb = np.asarray(he.convert("RGB"), dtype=np.uint8)
    palette = np.zeros((int(ids.max()) + 1, 3), dtype=np.uint8)
    for index, name in actual.items():
        palette[index] = (65, 186, 111) if predicted[index] == name else (226, 70, 81)
    output = rgb.copy()
    mask = ids > 0
    output[mask] = np.rint(rgb[mask] * 0.25 + palette[ids][mask] * 0.75).astype(np.uint8)
    return Image.fromarray(output)


def dapi_qc(ids: np.ndarray, cell_ids: np.ndarray, dapi: Image.Image) -> dict:
    if cell_ids.shape != ids.shape:
        raise ValueError("Nuclear and cell-proxy masks are not registered")
    values = np.asarray(dapi.resize((ids.shape[1], ids.shape[0]),
                                    Image.Resampling.BILINEAR), dtype=np.float32)
    inside = ids > 0
    outside = ~inside
    threshold = float(np.quantile(values, 0.9))
    mean_ratio = float(values[inside].mean() / max(values[outside].mean(), 1e-9))
    high_ratio = float(np.mean(values[inside] >= threshold) /
                       max(float(np.mean(values[outside] >= threshold)), 1e-9))
    ring = (cell_ids > 0) & ~inside
    size = int(max(ids.max(), cell_ids.max())) + 1
    nuclear_counts = np.bincount(ids[inside], minlength=size)
    ring_counts = np.bincount(cell_ids[ring], minlength=size)
    nuclear_mean = np.bincount(ids[inside], weights=values[inside], minlength=size) / np.maximum(nuclear_counts, 1)
    ring_mean = np.bincount(cell_ids[ring], weights=values[ring], minlength=size) / np.maximum(ring_counts, 1)
    eligible = (nuclear_counts > 0) & (ring_counts > 0)
    eligible[0] = False
    paired_ratio = nuclear_mean[eligible] / np.maximum(ring_mean[eligible], 1e-9)
    return {"nuclear_to_non_nuclear_mean_ratio": round(mean_ratio, 3),
            "top_decile_nuclear_enrichment": round(high_ratio, 3),
            "passes_enrichment_screen": high_ratio > 1.0,
            "paired_nuclei": int(eligible.sum()),
            "median_nucleus_to_local_ring_ratio": round(float(np.median(paired_ratio)), 3),
            "fraction_nuclei_brighter_than_local_ring": round(float(np.mean(paired_ratio > 1)), 3)}


def build(results: Path, site: Path, scores: Path | None = None) -> dict:
    scores = scores or results
    cohort = json.loads((results / "COHORT.json").read_text())
    report = json.loads((scores / "report.json").read_text())
    expert_rows = csv_rows(scores / "cell_table.csv")
    prediction_rows = csv_rows(scores / "predictions.csv")
    expert = {(r["sample_id"], int(r["label_id"])): r["cell_type"] for r in expert_rows}
    predicted = {(r["sample_id"], int(r["label_id"])): r["predicted_cell_type"]
                 for r in prediction_rows}
    if len(expert) != len(expert_rows) or len(predicted) != len(prediction_rows) or set(expert) != set(predicted):
        raise ValueError("Predictions and expert labels must align one-to-one")
    if not (results / "evidence" / "BUNDLE_COMPLETE.json").exists():
        raise ValueError("The bounded GigaTIME bundle has no completion marker")
    output = site / "assets" / "monusac-transfer"
    output.mkdir(parents=True, exist_ok=True)
    gallery = []
    for record in cohort["records"]:
        if record["split"] != "test":
            continue
        sample = record["sample_id"]
        source = results / "prepared" / sample
        generated = results / "output" / sample
        if not (generated / "RUN_COMPLETE.json").exists():
            raise ValueError(f"Missing complete inference case: {sample}")
        metadata = json.loads((generated / "gigatime_output" / "gigatime_metadata.json").read_text())
        if abs(metadata["source_mpp"] - record["mpp"]) > 1e-6 or abs(metadata["effective_mpp"] - 0.25) > 1e-6:
            raise ValueError(f"GigaTIME pixel resolution not calibrated: {sample}")
        with Image.open(source / "he.ome.tif") as image:
            he = image.convert("RGB")
        with Image.open(source / "expert_nuclei.ome.tif") as image:
            ids = np.asarray(image, dtype=np.int32)
        with Image.open(source / "expert_anchored_cell_proxy.ome.tif") as image:
            cell_ids = np.asarray(image, dtype=np.int32)
        actual = {index: name for (case, index), name in expert.items() if case == sample}
        prediction = {index: name for (case, index), name in predicted.items() if case == sample}
        if not actual or set(actual) != set(prediction):
            raise ValueError(f"Missing or mismatched predictions: {sample}")
        panels = {
            "he": he,
            "expert": color_overlay(he, ids, actual),
            "predicted": color_overlay(he, ids, prediction),
            "errors": error_overlay(he, ids, actual, prediction),
        }
        marker_root = generated / "gigatime_output" / "jpg_image"
        with Image.open(marker_root / "DAPI.jpg") as image:
            qc = dapi_qc(ids, cell_ids, image.convert("L"))
        for marker in ("DAPI", "CK", "CD3", "CD68"):
            with Image.open(marker_root / f"{marker}.jpg") as image:
                panels[marker.lower()] = image.convert("L")
        for name, image in panels.items():
            save_webp(image, output / f"{sample}-{name}.webp")
        confusion = [[sum(actual[index] == a and prediction[index] == p for index in actual)
                      for p in CLASSES] for a in CLASSES]
        support = Counter(actual.values())
        correct = sum(actual[index] == prediction[index] for index in actual)
        gallery.append({"sample_id": sample, "patient_id": record["patient_id"],
                        "source_mpp": record["mpp"], "model_mpp": 0.25,
                        "width": record["width"], "height": record["height"],
                        "n_cells": len(actual), "correct": correct,
                        "accuracy": round(correct / len(actual), 5),
                        "support": {name: support[name] for name in CLASSES},
                        "confusion_matrix": confusion, "dapi_qc": qc})
    if len(gallery) != 10 or sum(item["n_cells"] for item in gallery) != report["test_cells"]:
        raise ValueError("Gallery does not cover the frozen test cohort")
    summed_confusion = np.sum([item["confusion_matrix"] for item in gallery], axis=0)
    if not np.array_equal(summed_confusion, np.asarray(report["test"]["confusion_matrix"])):
        raise ValueError("Gallery class counts disagree with the saved fastPLS evaluation")
    summary = {"method": report["method"], "cohort": "MoNuSAC lung training to prostate test",
               "train_images": sum(r["split"] == "train" for r in cohort["records"]),
               "test_images": len(gallery), "train_patients": len(report["train_patients"]),
               "test_patients": len(report["test_patients"]),
               "train_cells": report["train_cells"], "test_cells": report["test_cells"],
               "test": report["test"], "test_equal_patient_accuracy": report["test_by_patient"]["equal_weight_accuracy"],
               "majority_baseline": report["majority_baseline"],
               "dapi_qc_passed_test_images": sum(r["dapi_qc"]["passes_enrichment_screen"] for r in gallery),
               "class_order": CLASSES,
               "scope": "Cell classification conditional on expert nuclei; 3 µm nucleus-anchored cell proxy. Virtual markers are not measured protein.",
               "cases": gallery}
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--scores", type=Path, help="Model/score directory, if separate from the image bundle")
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.results, args.site, args.scores)
    print(json.dumps({"test_images": result["test_images"], "test_cells": result["test_cells"],
                      "accuracy": result["test"]["accuracy"],
                      "balanced_accuracy": result["test"]["balanced_accuracy"],
                      "dapi_qc_passed_test_images": result["dapi_qc_passed_test_images"]}, indent=2))
