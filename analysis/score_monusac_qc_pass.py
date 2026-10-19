"""Evaluate corrected MoNuSAC cases that pass prespecified native DAPI QC.

This is a conditional, QC-selected expert-region analysis. The 120-image
cohort and patient split remain frozen; failed-QC images are counted and omitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def complete_quality_pass(report: dict) -> bool:
    """Require explicit passes for nuclei, border shading and spatial shading."""
    return all(report.get(field) == "pass" for field in
               ("status", "alignment_status", "border_shadow_status",
                "spatial_shadow_status"))


def run(source: Path, output: Path, tambua: Path) -> dict:
    from tambua.model import metrics

    if not (source / "evidence/BUNDLE_COMPLETE.json").is_file() or \
            (source / "evidence/failure.txt").exists():
        raise ValueError("Corrected inference bundle did not finish cleanly")
    records = json.loads((source / "input/COHORT.json").read_text())["records"]
    if len(records) != 120 or sum(r["split"] == "train" for r in records) != 88 or \
            sum(r["split"] == "test" for r in records) != 32:
        raise ValueError("Frozen cohort changed")
    status = {}
    qc_reasons = Counter()
    for row in records:
        sample = row["sample_id"]
        case = source / "output" / sample
        if not (case / "RUN_COMPLETE.json").is_file():
            raise ValueError(f"Incomplete corrected inference: {sample}")
        validation = json.loads((case / "VALIDATION.json").read_text())
        if validation["source_color_reference"] != "scanner_svs_rgb" or \
                abs(validation["effective_mpp"] - 0.25) > 1e-4:
            raise ValueError(f"Uncorrected image or physical scale: {sample}")
        qc_path = case / "gigatime_output/output_gigatime_nuclear_qc.json"
        if not qc_path.is_file():
            raise ValueError(f"Native DAPI/shadow QC missing: {sample}")
        qc = json.loads(qc_path.read_text())
        if qc.get("status") != validation["nuclear_qc"]:
            raise ValueError(f"Native QC and validated status disagree: {sample}")
        status[sample] = {key: qc.get(key) for key in
                          ("status", "alignment_status", "border_shadow_status",
                           "spatial_shadow_status")}
        if not complete_quality_pass(qc):
            for key, value in status[sample].items():
                if value != "pass":
                    qc_reasons[f"{key}:{value}"] += 1
    selected = [row for row in records if complete_quality_pass(status[row["sample_id"]])]
    training = [r for r in selected if r["split"] == "train"]
    testing = [r for r in selected if r["split"] == "test"]
    train_organs = {r["organ"] for r in training}
    test_organs = {r["organ"] for r in testing}
    if not training or not testing or not test_organs <= train_organs:
        raise ValueError("Passing test tissues lack passing training images")
    if {r["patient_id"] for r in training} & {r["patient_id"] for r in testing}:
        raise ValueError("QC selection broke patient separation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=("sample_id", "patient_id", "organ",
                                                     "split", "labels_csv", "nuclei_csv", "cell_csv"))
        writer.writeheader()
        for row in selected:
            sample = row["sample_id"]
            quant = source / "output" / sample / "quantification"
            writer.writerow(dict(sample_id=sample, patient_id=row["patient_id"],
                                 organ=row["organ"], split=row["split"],
                                 labels_csv=source / "prepared" / sample / "expert_types.csv",
                                 nuclei_csv=quant / "output_nuclei_gigatime_mean_intensity.csv",
                                 cell_csv=quant / "output_cell_gigatime_mean_intensity.csv"))
    env = dict(os.environ, PYTHONPATH=str(tambua / "src"))

    def call(*args: str | Path) -> None:
        subprocess.run([sys.executable, "-m", "tambua.cli", *map(str, args)],
                       cwd=tambua, env=env, check=True)

    table = output / "cell_table.csv"
    model = output / "classifier.rds"
    report_path = output / "report.json"
    predictions = output / "predictions.csv"
    call("build-table", "--manifest", manifest, "--output", table,
         "--compartments", "nuclei-cell")
    call("train", "--table", table, "--model", model, "--report", report_path)
    call("predict", "--table", table, "--model", model, "--output", predictions)
    report = json.loads(report_path.read_text())
    with table.open(newline="", encoding="utf-8") as stream:
        expert = {(r["sample_id"], r["label_id"]): r for r in csv.DictReader(stream)}
    with predictions.open(newline="", encoding="utf-8") as stream:
        predicted = {(r["sample_id"], r["label_id"]): r["predicted_cell_type"]
                     for r in csv.DictReader(stream)}
    if set(expert) != set(predicted):
        raise ValueError("Prediction and expert IDs differ")
    classes = report["test"]["classes"]
    by_organ = {}
    for organ in sorted(test_organs):
        keys = [key for key, row in expert.items()
                if row["split"] == "test" and row["organ"] == organ]
        by_organ[organ] = metrics(np.asarray([expert[key]["cell_type"] for key in keys]),
                                  np.asarray([predicted[key] for key in keys]), classes)
    report.update(qc_selection={"all_images": len(records), "pass_images": len(selected),
                                "train_pass": len(training), "test_pass": len(testing),
                                "train_fail": 88 - len(training), "test_fail": 32 - len(testing),
                                "pass_by_split_organ": {
                                    f"{split}:{organ}": count for (split, organ), count in
                                    sorted(Counter((r["split"], r["organ"]) for r in selected).items())},
                                "selected_test_patients": len({r["patient_id"] for r in testing}),
                                "excluded_status_counts": dict(sorted(qc_reasons.items())),
                                "rule": "All native status, alignment, border-shadow and spatial-shadow fields must explicitly pass"},
                  test_by_tissue=by_organ)
    report["claim_limit"] += (
        " QC-selected evaluation at expert-drawn regions, not end-to-end StarDist detection. "
        "Selection on virtual-DAPI quality can shift the tissue/cell distribution; "
        "the source annotation geometry also requires review."
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (output / "qc_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tambua", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tambua.resolve() / "src"))
    result = run(args.source.resolve(), args.output.resolve(), args.tambua.resolve())
    print(json.dumps({"qc_selection": result["qc_selection"], "test": result["test"]}, indent=2))
