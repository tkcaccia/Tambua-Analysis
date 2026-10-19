#!/usr/bin/env python3
"""Score the frozen four-tissue MoNuSAC cohort after all GPU cases complete.

The 88 training and 32 test ROIs are patient-disjoint, and every test tissue
occurs in training. This script is CPU-only and refuses incomplete bundles.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.report_class_coverage import report as class_coverage_report


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def case_qc_status(folder: Path, sample: str) -> str:
    report_path = next((p for p in (
        folder / "gigatime_output" / f"{sample}_gigatime_nuclear_qc.json",
        folder / "gigatime_output" / "output_gigatime_nuclear_qc.json",
    ) if p.is_file()), None)
    summary_path = next((p for p in (
        folder / "quantification" / f"{sample}_gigatime_integrated_quantification_summary.json",
        folder / "quantification" / "output_gigatime_integrated_quantification_summary.json",
    ) if p.is_file()), None)
    if report_path is None or summary_path is None:
        return "missing"
    report = json.loads(report_path.read_text())
    summary = json.loads(summary_path.read_text())
    return "pass" if report.get("status") == "pass" and summary.get("technical_qc_passed") is True else "fail"


def require_scanner_rgb(records: list[dict], historical_unvalidated: bool) -> None:
    if not historical_unvalidated and any(
        row.get("source_color_reference") != "scanner_svs_rgb" for row in records
    ):
        raise ValueError("MoNuSAC marker scoring requires scanner-SVS-corrected RGB inputs")


def score(source: Path, destination: Path, tambua: Path,
          *, historical_unvalidated: bool = False) -> dict:
    from tambua.model import metrics

    cohort = json.loads((source / "input/COHORT.json").read_text())
    records = cohort["records"]
    if len(records) != 120 or sum(r["split"] == "train" for r in records) != 88 or \
            sum(r["split"] == "test" for r in records) != 32:
        raise ValueError("Frozen 88/32 cohort changed")
    if not (source / "evidence/BUNDLE_COMPLETE.json").is_file() or \
            (source / "evidence/failure.txt").exists():
        raise ValueError("GPU bundle did not complete cleanly")
    train_patients = {r["patient_id"] for r in records if r["split"] == "train"}
    test_patients = {r["patient_id"] for r in records if r["split"] == "test"}
    if train_patients & test_patients or len(train_patients) != 40 or len(test_patients) != 19:
        raise ValueError("Unexpected patient split")
    organs = {r["organ"] for r in records if r["split"] == "test"}
    if organs != {"breast", "kidney", "lung", "prostate"} or not organs <= \
            {r["organ"] for r in records if r["split"] == "train"}:
        raise ValueError("A test tissue is absent from training")
    require_scanner_rgb(records, historical_unvalidated)
    qc_statuses = []
    for row in records:
        folder = source / "output" / row["sample_id"]
        if not (folder / "RUN_COMPLETE.json").is_file() or \
                not (folder / "VALIDATION.json").is_file():
            raise ValueError(f"Unverified image: {row['sample_id']}")
        status = case_qc_status(folder, row["sample_id"])
        if status == "fail" or (status == "missing" and not historical_unvalidated):
            raise ValueError(f"GigaTIME nuclear/shadow QC is {status}: {row['sample_id']}")
        qc_statuses.append(status)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to overwrite model or scores")
    destination.mkdir(parents=True, exist_ok=True)
    sample_organ = {r["sample_id"]: r["organ"] for r in records}
    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("sample_id", "patient_id", "organ",
                                                    "split", "labels_csv", "nuclei_csv", "cell_csv"))
        writer.writeheader()
        for row in records:
            sample = row["sample_id"]
            quant = source / "output" / sample / "quantification"
            writer.writerow({"sample_id": sample, "patient_id": row["patient_id"],
                             "organ": row["organ"], "split": row["split"],
                             "labels_csv": source / "prepared" / sample / "expert_types.csv",
                             "nuclei_csv": quant / "output_nuclei_gigatime_mean_intensity.csv",
                             "cell_csv": quant / "output_cell_gigatime_mean_intensity.csv"})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tambua / "src")
    def call(*args: str | Path) -> None:
        subprocess.run([sys.executable, "-m", "tambua.cli", *map(str, args)],
                       cwd=tambua, env=env, check=True)
    table = destination / "cell_table.csv"
    model = destination / "classifier.rds"
    report_path = destination / "report.json"
    predictions = destination / "predictions.csv"
    call("build-table", "--manifest", manifest, "--output", table,
         "--compartments", "nuclei-cell")
    call("train", "--table", table, "--model", model, "--report", report_path)
    call("predict", "--table", table, "--model", model, "--output", predictions)
    report = json.loads(report_path.read_text())
    expert = {(r["sample_id"], r["label_id"]): r for r in csv_rows(table)}
    predicted = {(r["sample_id"], r["label_id"]): r["predicted_cell_type"]
                 for r in csv_rows(predictions)}
    if set(expert) != set(predicted) or len(expert) != len(csv_rows(table)):
        raise ValueError("Cell predictions and expert annotations are not aligned")
    classes = report["test"]["classes"]
    per_organ = {}
    for organ in sorted(organs):
        keys = [key for key, row in expert.items()
                if row["split"] == "test" and sample_organ[key[0]] == organ]
        actual = np.asarray([expert[key]["cell_type"] for key in keys])
        values = np.asarray([predicted[key] for key in keys])
        item = metrics(actual, values, classes)
        item["images"] = sum(r["organ"] == organ and r["split"] == "test" for r in records)
        item["patients"] = len({r["patient_id"] for r in records
                                if r["organ"] == organ and r["split"] == "test"})
        per_organ[organ] = item
    if sum(x["n"] for x in per_organ.values()) != report["test_cells"]:
        raise ValueError("Tissue-stratified scores do not cover every test cell")
    report["test_by_tissue"] = per_organ
    image_cases = []
    for row in records:
        if row["split"] != "test":
            continue
        keys = [key for key in expert if key[0] == row["sample_id"]]
        item = metrics(np.asarray([expert[key]["cell_type"] for key in keys]),
                       np.asarray([predicted[key] for key in keys]), classes)
        image_cases.append({"sample_id": row["sample_id"], "organ": row["organ"],
                            "n_cells": item["n"], "support": item["support"],
                            "confusion_matrix": item["confusion_matrix"]})
    report["test_image_class_coverage"] = class_coverage_report(image_cases, classes)
    report["cohort"] = {"train_images": 88, "test_images": 32,
                        "train_patients": 40, "test_patients": 19,
                        "tissues_in_both_splits": sorted(organs),
                        "source_cohort_sha256": __import__("hashlib").sha256(
                            (source / "input/COHORT.json").read_bytes()).hexdigest()}
    report["gigatime_quality_status"] = (
        "historical_unvalidated" if "missing" in qc_statuses else "native_nuclear_qc_passed"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tambua", type=Path, required=True)
    parser.add_argument("--historical-unvalidated", action="store_true",
                        help="Reproduce archived pre-QC scores only; marks the report unvalidated")
    args = parser.parse_args()
    sys.path.insert(0, str(args.tambua.resolve() / "src"))
    report = score(args.source.resolve(), args.output.resolve(), args.tambua.resolve(),
                   historical_unvalidated=args.historical_unvalidated)
    print(json.dumps({"test": report["test"], "test_by_tissue": report["test_by_tissue"]}, indent=2))
