#!/usr/bin/env python3
"""Fit fastPLS 0.4 on QC-passing detected cells, then predict every test cell.

The fitted model uses only conservatively matched StarDist cells from the
frozen fit patients. It never sees development or organizer-test labels during
fitting. The old source-polygon model and QC are not used in this analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This scoring process must explicitly hide all GPUs")


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def complete_qc(row: dict) -> bool:
    return all(value == "pass" for value in row["stardist_nuclear_qc"].values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-summary", type=Path, required=True)
    ap.add_argument("--original-quant", type=Path, required=True)
    ap.add_argument("--additional-quant", type=Path, required=True)
    ap.add_argument("--tambua-src", type=Path, required=True)
    ap.add_argument("--r-library", type=Path,
                    help="Isolated library containing GitHub fastPLS 0.4")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    qc = json.loads(args.qc_summary.read_text())
    if qc.get("cases") != 120 or qc.get("status") != "complete":
        raise ValueError("StarDist-reference QC is incomplete")
    selected = [row for row in qc["records"] if complete_qc(row)]
    if Counter(row["split"] for row in selected) != \
            {"fit": 33, "development": 4, "test": 10}:
        raise ValueError("Frozen corrected-QC cohort changed")
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = []
    for row in selected:
        sample = row["sample_id"]
        folder = next((root / sample for root in
                       (args.original_quant, args.additional_quant)
                       if (root / sample / "CASE_COMPLETE.json").exists()), None)
        if folder is None:
            raise FileNotFoundError(f"No detected-cell quantification: {sample}")
        checkpoint = json.loads((folder / "CASE_COMPLETE.json").read_text())
        if checkpoint["sample_id"] != sample or checkpoint["status"] != "complete" or \
                checkpoint["source_color_reference"] != "scanner_svs_rgb" or \
                abs(float(checkpoint["gigatime_effective_mpp"]) - .25) > 1e-4:
            raise ValueError(f"Quantification identity or calibration failed: {sample}")
        for name in ("nuclei", "cell"):
            summary = json.loads((folder / f"detected_{name}_summary.json").read_text())
            if len(summary["channel_names"]) != 23 or \
                    summary["objects_quantified"] != checkpoint["detected_cells_quantified"]:
                raise ValueError(f"Incomplete 23-channel detected-cell table: {sample}")
        cases.append((row, folder, checkpoint))
    args.output.mkdir(parents=True)

    def write_manifest(path: Path, source: list, *, labelled: bool) -> None:
        columns = ("sample_id", "patient_id", "organ", "split", "labels_csv",
                   "nuclei_csv", "cell_csv") if labelled else \
                  ("sample_id", "patient_id", "nuclei_csv", "cell_csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row, folder, _checkpoint in source:
                item = {"sample_id": row["sample_id"], "patient_id": row["patient_id"],
                        "nuclei_csv": folder / "detected_nuclei_mean_intensity.csv",
                        "cell_csv": folder / "detected_cell_mean_intensity.csv"}
                if labelled:
                    item.update(organ=row["organ"],
                                split="train" if row["split"] == "fit" else "test",
                                labels_csv=folder / "expert_types_on_detected.csv")
                writer.writerow(item)

    labelled = [case for case in cases if case[0]["split"] in ("fit", "test")]
    test = [case for case in cases if case[0]["split"] == "test"]
    write_manifest(args.output / "labelled_manifest.csv", labelled, labelled=True)
    write_manifest(args.output / "test_inference_manifest.csv", test, labelled=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.tambua_src.resolve())
    if args.r_library:
        env["TAMBUA_R_LIB"] = str(args.r_library.resolve())
    def cli(*values: str) -> None:
        subprocess.run([sys.executable, "-m", "tambua.cli", *values],
                       check=True, env=env, stdout=subprocess.DEVNULL)
    table = args.output / "matched_cells.csv"
    model = args.output / "detected_cell_classifier.rds"
    report = args.output / "model_report.json"
    cli("build-table", "--manifest", str(args.output / "labelled_manifest.csv"),
        "--output", str(table), "--compartments", "nuclei-cell")
    cli("train", "--table", str(table), "--model", str(model),
        "--report", str(report))
    matched_predictions = args.output / "matched_cell_predictions.csv"
    cli("predict", "--table", str(table), "--model", str(model),
        "--output", str(matched_predictions))
    unlabelled_table = args.output / "all_detected_test_cells.csv"
    cli("build-inference", "--manifest", str(args.output / "test_inference_manifest.csv"),
        "--output", str(unlabelled_table), "--compartments", "nuclei-cell")
    all_predictions = args.output / "all_detected_test_predictions.csv"
    cli("predict", "--table", str(unlabelled_table), "--model", str(model),
        "--output", str(all_predictions))
    model_report = json.loads(report.read_text())
    if model_report["fastpls_version"] != "0.4" or \
            len(model_report["feature_names"]) != 46:
        raise ValueError("Wrong fastPLS version or feature panel")
    matched_rows = csv_rows(table)
    all_rows = csv_rows(unlabelled_table)
    predictions = csv_rows(all_predictions)
    if len(all_rows) != len(predictions) or \
            {(r["sample_id"], r["label_id"]) for r in all_rows} != \
            {(r["sample_id"], r["label_id"]) for r in predictions}:
        raise ValueError("Detected-cell prediction IDs differ from marker-table IDs")
    if len({r["patient_id"] for r in matched_rows if r["split"] == "train"} &
           {r["patient_id"] for r in matched_rows if r["split"] == "test"}):
        raise ValueError("Patient leakage")
    by_image = Counter(r["sample_id"] for r in predictions)
    result = {"status": "complete", "device": "CPU",
              "method": "fastPLS 0.4 SIMPLS-LDA; 23 GigaTIME channels in each of nucleus and detected-cell proxy",
              "qc_reference": "StarDist H&E nuclei; DAPI/border/spatial thresholds unchanged",
              "qc_summary_sha256": digest(args.qc_summary),
              "fit_images": 33, "development_images_not_used": 4,
              "organizer_test_images": 10,
              "matched_fit_cells": sum(row["split"] == "train" for row in matched_rows),
              "matched_test_cells": sum(row["split"] == "test" for row in matched_rows),
              "all_detected_test_cells_predicted": len(predictions),
              "test_detected_cells_by_image": dict(sorted(by_image.items())),
              "test_metrics_on_conservatively_matched_cells": model_report["test"],
              "interpretation": ("Matched-cell classification conditional on H&E StarDist detection, "
                                 "not full detection precision. Organizer test has no matched neutrophils; "
                                 "four-class balanced accuracy is undefined. This retrospective test set has "
                                 "already been used for pipeline auditing, so it is not a pristine final holdout.")}
    staged = args.output / "DETECTED_CELL_STUDY_COMPLETE.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(args.output / "DETECTED_CELL_STUDY_COMPLETE.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
