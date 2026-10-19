#!/usr/bin/env python3
"""Reproduce the CPU-only fastPLS 0.4 lung-to-prostate evaluation.

The calibrated CellPhenotyper/GigaTIME outputs and prepared expert labels must
already exist. This script never starts or resumes GPU inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def score(source: Path, destination: Path, tambua: Path) -> dict:
    cohort = json.loads((source / "COHORT.json").read_text())
    if len(cohort["records"]) != 37 or not (source / "evidence" / "BUNDLE_COMPLETE.json").exists():
        raise ValueError("Expected a completed, frozen 37-image GigaTIME bundle")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Refusing to overwrite a prior model or score")
    destination.mkdir(parents=True, exist_ok=True)
    fields = ("sample_id", "patient_id", "split", "labels_csv", "nuclei_csv", "cell_csv")
    manifest = destination / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in cohort["records"]:
            sample = record["sample_id"]
            if not (source / "output" / sample / "RUN_COMPLETE.json").exists():
                raise ValueError(f"Unverified GigaTIME image: {sample}")
            quant = source / "output" / sample / "quantification"
            writer.writerow({"sample_id": sample, "patient_id": record["patient_id"],
                             "split": record["split"],
                             "labels_csv": source / "prepared" / sample / "expert_types.csv",
                             "nuclei_csv": quant / "output_nuclei_gigatime_mean_intensity.csv",
                             "cell_csv": quant / "output_cell_gigatime_mean_intensity.csv"})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tambua / "src")
    def call(*args: str) -> None:
        subprocess.run([sys.executable, "-m", "tambua.cli", *map(str, args)],
                       cwd=tambua, env=env, check=True)
    table = destination / "cell_table.csv"
    model = destination / "classifier.rds"
    report_path = destination / "report.json"
    prediction = destination / "predictions.csv"
    call("build-table", "--manifest", manifest, "--output", table,
         "--compartments", "nuclei-cell")
    call("train", "--table", table, "--model", model, "--report", report_path)
    call("predict", "--table", table, "--model", model, "--output", prediction)
    report = json.loads(report_path.read_text())
    if report["test_cells"] != 3584 or report["train_cells"] != 5243:
        raise ValueError("The frozen cohort no longer matches its recorded cell counts")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Read-only completed GigaTIME and expert-label bundle")
    parser.add_argument("--output", type=Path, required=True,
                        help="New empty directory for model and evaluation")
    parser.add_argument("--tambua", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.source.resolve(), args.output.resolve(), args.tambua.resolve())
    print(json.dumps({"test": report["test"], "fit_seconds": report["fit_seconds"]}, indent=2))
