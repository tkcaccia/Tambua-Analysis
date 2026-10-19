#!/usr/bin/env python3
"""Replace accepted-case overlays with predictions from the shadow-strict fit.

All original case previews are copied to a new gallery first. Failed or
insufficient-QC cases retain an explicitly diagnostic all-image model output.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from build_monusac_gallery import CLASSES, save_webp
from build_monusac_matched_gallery import overlay


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def accepted(qc: dict) -> bool:
    return all(qc.get(key) == "pass" for key in
               ("status", "alignment_status", "border_shadow_status",
                "spatial_shadow_status"))


def merge(source: Path, scores: Path, old_test: Path, old_train: Path,
          destination: Path) -> dict:
    if not (source / "evidence/BUNDLE_COMPLETE.json").is_file():
        raise ValueError("Corrected 120-image bundle is incomplete")
    if destination.exists():
        raise FileExistsError(destination)
    report = json.loads((scores / "report.json").read_text())
    expert_rows = csv_rows(scores / "cell_table.csv")
    predicted_rows = csv_rows(scores / "predictions.csv")
    expert = {(r["sample_id"], int(r["label_id"])): r["cell_type"] for r in expert_rows}
    predicted = {(r["sample_id"], int(r["label_id"])): r["predicted_cell_type"]
                 for r in predicted_rows}
    if len(expert) != len(expert_rows) or set(expert) != set(predicted):
        raise ValueError("Strict prediction and expert cell IDs differ")
    included = {sample for sample, _ in expert}
    cases_by_split = {}
    for split, origin, folder in (("test", old_test, "monusac-corrected"),
                                  ("train", old_train, "monusac-corrected-train")):
        target = destination / "assets" / folder
        shutil.copytree(origin, target)
        manifest_path = target / "results.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["split"] = split
        manifest["cohort"] = "MoNuSAC four-tissue patient-disjoint cohort"
        cases_by_split[split] = manifest["cases"]
        for case in manifest["cases"]:
            sample = case["sample_id"]
            case["split"] = split
            qc = case["native_nuclear_qc"]
            case["accepted_qc"] = accepted(qc)
            if case["accepted_qc"] != (sample in included):
                raise ValueError(f"Strict score and full QC disagree: {sample}")
            if not case["accepted_qc"]:
                case["model_scope"] = "diagnostic_all_image_fit_failed_or_insufficient_qc"
                continue
            prepared = source / "prepared" / sample
            with Image.open(prepared / "he.ome.tif") as image:
                he = image.convert("RGB")
            with Image.open(prepared / "expert_nuclei.ome.tif") as image:
                ids = np.asarray(image, dtype=np.int32)
            reference = {i: name for (name_sample, i), name in expert.items()
                         if name_sample == sample}
            values = {i: name for (name_sample, i), name in predicted.items()
                      if name_sample == sample}
            if set(np.unique(ids)) - {0} != set(reference):
                raise ValueError(f"Strict predictions do not cover expert polygons: {sample}")
            save_webp(overlay(he, ids, values), target / f"{sample}-predicted.webp")
            save_webp(overlay(he, ids, values, actual=reference),
                      target / f"{sample}-errors.webp")
            case["correct"] = sum(reference[i] == values[i] for i in reference)
            case["accuracy"] = round(case["correct"] / len(reference), 5)
            case["n_cells"] = len(reference)
            case["confusion_matrix"] = [
                [sum(reference[i] == actual and values[i] == prediction
                     for i in reference) for prediction in CLASSES]
                for actual in CLASSES]
            case["model_scope"] = "strict_qc_fastpls_fit"
        manifest["strict_qc_selection"] = report["qc_selection"]
        manifest["accepted_qc_images_in_split"] = sum(c["accepted_qc"]
                                                       for c in manifest["cases"])
        if split == "test":
            manifest["all_image_diagnostic_test"] = manifest.pop("test")
            manifest["strict_qc_test"] = report["test"]
            confusion = np.sum([c["confusion_matrix"] for c in manifest["cases"]
                                if c["accepted_qc"]], axis=0)
            if not np.array_equal(confusion, np.asarray(report["test"]["confusion_matrix"])):
                raise ValueError("Strict gallery and test score disagree")
        else:
            for unrelated in ("test", "test_by_tissue", "test_equal_patient_accuracy",
                              "majority_baseline"):
                manifest.pop(unrelated, None)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if len(cases_by_split["train"]) != 88 or len(cases_by_split["test"]) != 32:
        raise ValueError("Gallery does not cover all 120 images")
    return {"strict_train_images": sum(c["accepted_qc"] for c in cases_by_split["train"]),
            "strict_test_images": sum(c["accepted_qc"] for c in cases_by_split["test"]),
            "strict_test_cells": report["test_cells"],
            "strict_test_accuracy": report["test"]["accuracy"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--old-test", type=Path, required=True)
    parser.add_argument("--old-train", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.scores, args.old_test,
                           args.old_train, args.destination), indent=2))
