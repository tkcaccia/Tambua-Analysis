#!/usr/bin/env python3
"""Evaluate expanded QC-passing MoNuSAC cells with GitHub fastPLS 0.4.

The original organizer test is retrospective. The additional patient-held-out
test images are the primary evaluation; development images select no final
test labels. All 23 GigaTIME channels in both nuclear and cell compartments
enter the classifier. Nothing is scored until the 93-image inference and
independent StarDist quantification bundles are complete.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plan(old_qc: dict, expansion: dict) -> list[dict]:
    if old_qc.get("status") != "complete" or old_qc.get("cases") != 120:
        raise ValueError("Original corrected StarDist QC summary is incomplete")
    prior = [dict(sample_id=row["sample_id"], patient_id=row["patient_id"],
                  organ=row["organ"], split={"fit": "train", "development": "development",
                                                  "test": "historical_test"}[row["split"]],
                  source="original")
             for row in old_qc["records"]
             if all(value == "pass" for value in row["stardist_nuclear_qc"].values())]
    if len(prior) != 47 or Counter(row["split"] for row in prior) != \
            {"train": 33, "development": 4, "historical_test": 10}:
        raise ValueError("The 47-image corrected QC cohort changed")
    if expansion.get("status") != "complete" or expansion.get("device") != "CPU" or \
            expansion.get("source_images") != 93 or \
            expansion.get("qc_passing_images") != len(expansion.get("records", [])):
        raise ValueError("Complete independent 93-image quantification is required")
    prior_test_patients = {row["patient_id"] for row in old_qc["records"]
                           if row["split"] == "test"}
    added = [dict(sample_id=row["sample_id"], patient_id=row["patient_id"],
                  organ=row["organ"], split=row["split"], source="expansion",
                  patient_seen_in_original_test=(row["patient_id"] in prior_test_patients)
                  if row["split"] == "test" else False)
             for row in expansion["records"]]
    cases = prior + added
    if len({row["sample_id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate image between original and expansion")
    permitted = {"train", "development", "test", "historical_test"}
    if any(row["split"] not in permitted for row in cases):
        raise ValueError("Unknown patient split")
    by_patient: dict[str, set[str]] = defaultdict(set)
    for row in cases:
        family = "test" if row["split"] in {"test", "historical_test"} else row["split"]
        by_patient[row["patient_id"]].add(family)
    if any(len(splits) != 1 for splits in by_patient.values()):
        raise ValueError("Patient leakage between fitting, development or test")
    if not any(row["split"] == "test" for row in added):
        raise ValueError("No new held-out test image passed GigaTIME QC")
    if not any(row["split"] == "train" for row in added):
        raise ValueError("No expanded training image passed GigaTIME QC")
    if {row["organ"] for row in cases if row["split"] == "test"} - \
            {row["organ"] for row in cases if row["split"] == "train"}:
        raise ValueError("New test tissue absent from QC-passing training")
    return cases


def manifest(cases: list[dict], old_root: Path, new_root: Path,
             destination: Path, selected: set[str], *, labelled: bool) -> None:
    fields = (["sample_id", "patient_id", "organ", "split", "labels_csv"] if labelled
              else ["sample_id", "patient_id"])
    fields += ["nuclei_csv", "cell_csv"]
    chosen = [row for row in cases if row["split"] in selected]
    if not chosen:
        raise ValueError(f"No cases selected for {destination}")
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            folder = (old_root if row["source"] == "original" else new_root) / row["sample_id"]
            checkpoint = json.loads((folder / "CASE_COMPLETE.json").read_text())
            if checkpoint.get("status") != "complete" or \
                    checkpoint.get("sample_id") != row["sample_id"] or \
                    checkpoint.get("source_color_reference") != "scanner_svs_rgb" or \
                    abs(float(checkpoint.get("gigatime_effective_mpp", -1)) - .25) > 1e-4:
                raise ValueError(f"Missing or mismatched cell quantification: {row['sample_id']}")
            item = dict(sample_id=row["sample_id"], patient_id=row["patient_id"],
                        nuclei_csv=folder / "detected_nuclei_mean_intensity.csv",
                        cell_csv=folder / "detected_cell_mean_intensity.csv")
            if labelled:
                item.update(organ=row["organ"],
                            split="train" if row["split"] == "train" else "test",
                            labels_csv=folder / "expert_types_on_detected.csv")
            if any(not Path(value).is_file() for key, value in item.items()
                   if key.endswith("_csv")):
                raise FileNotFoundError(f"Missing full-channel table: {row['sample_id']}")
            writer.writerow(item)


def matched_metrics(table: Path, predicted: Path, classes: list[str],
                    *, only_patients: set[str] | None = None) -> dict:
    from tambua.model import metrics
    labels = rows(table)
    predictions = rows(predicted)
    if only_patients is not None:
        labels = [row for row in labels if row["patient_id"] in only_patients]
        predictions = [row for row in predictions if row["patient_id"] in only_patients]
        if not labels:
            return {"status": "no_matched_cells_from_previously_unseen_test_patients"}
    actual = {(r["sample_id"], r["label_id"]): r for r in labels}
    called = {(r["sample_id"], r["label_id"]): r["predicted_cell_type"]
              for r in predictions}
    if len(actual) != len(labels) or len(called) != len(predictions) or \
            set(actual) != set(called):
        raise ValueError("Expert and predicted detected-cell IDs differ")
    y = np.asarray([r["cell_type"] for r in labels])
    pred = np.asarray([called[(r["sample_id"], r["label_id"])] for r in labels])
    pooled = metrics(y, pred, classes)
    organs = sorted({r["organ"] for r in labels})
    by_organ = {organ: metrics(y[[i for i, r in enumerate(labels) if r["organ"] == organ]],
                                pred[[i for i, r in enumerate(labels) if r["organ"] == organ]],
                                classes)
                for organ in organs}
    patients = sorted({r["patient_id"] for r in labels})
    patient_indices = {patient: np.asarray([i for i, row in enumerate(labels)
                                           if row["patient_id"] == patient], dtype=np.int64)
                       for patient in patients}
    per_class_patient = {}
    for name in classes:
        recalls = []
        for patient in patients:
            idx = patient_indices[patient]
            selected = y[idx] == name
            if np.any(selected):
                recalls.append(float(np.mean(pred[idx][selected] == name)))
        per_class_patient[name] = {"patients_with_expert_class": len(recalls),
                                   "equal_patient_recall": float(np.mean(recalls))
                                   if recalls else None}
    patient_macro = [per_class_patient[name]["equal_patient_recall"]
                     for name in classes]
    rng = np.random.default_rng(20261020)
    bootstrap = []
    for draw in rng.choice(patients, size=(2000, len(patients)), replace=True):
        index = np.concatenate([patient_indices[patient] for patient in draw])
        score = metrics(y[index], pred[index], classes)["balanced_accuracy"]
        if score is not None:
            bootstrap.append(score)
    return {"pooled": pooled, "by_organ": by_organ, "patients": len(patients),
            "equal_patient_per_class_recall": per_class_patient,
            "equal_patient_balanced_accuracy": float(np.mean(patient_macro))
                if all(value is not None for value in patient_macro) else None,
            "patient_bootstrap_valid_draws": len(bootstrap),
            "balanced_accuracy_95pct_patient_bootstrap":
                [float(x) for x in np.quantile(bootstrap, [.025, .975])]
                if bootstrap else None,
            "interpretation": "Classification conditional on independently detected, conservatively matched nuclei. A class absent from test makes four-class balanced accuracy undefined; a bootstrap draw missing a class is excluded and its count reported."}


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Expanded classifier scoring must hide GPUs")
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.old_qc.read_text())
    expanded_path = args.new_quant / "COHORT_COMPLETE.json"
    expanded = json.loads(expanded_path.read_text())
    cases = plan(old, expanded)
    sys.path.insert(0, str(args.tambua_src.resolve()))
    args.output.mkdir(parents=True)
    split_counts = Counter(row["split"] for row in cases)
    (args.output / "COHORT_PLAN.json").write_text(json.dumps({
        "old_qc_sha256": sha(args.old_qc), "expansion_quant_sha256": sha(expanded_path),
        "images_by_split": dict(split_counts),
        "patients_by_split": {split: len({row["patient_id"] for row in cases
                                           if row["split"] == split})
                              for split in sorted(split_counts)},
        "evaluation": "Additional images from organizer-test patients held out from fitting. Some patients were already audited in the original organizer test; report that overlap and an unseen-patient subset separately.",
        "additional_test_images_from_previously_seen_test_patients": sum(
            row["split"] == "test" and row["patient_seen_in_original_test"] for row in cases),
        "additional_test_images_from_unseen_test_patients": sum(
            row["split"] == "test" and not row["patient_seen_in_original_test"] for row in cases),
        "features": "All 23 native GigaTIME channels in independent StarDist nucleus and calibrated detected-cell proxy",
        "primary_metric": "Pooled four-class balanced accuracy conditional on conservative expert match",
        "records": cases}, indent=2) + "\n")
    manifest(cases, args.old_quant, args.new_quant,
             args.output / "fit_and_development_manifest.csv",
             {"train", "development"}, labelled=True)
    manifest(cases, args.old_quant, args.new_quant,
             args.output / "new_test_manifest.csv", {"test"}, labelled=True)
    manifest(cases, args.old_quant, args.new_quant,
             args.output / "new_test_all_detected_manifest.csv", {"test"}, labelled=False)
    env = dict(os.environ, PYTHONPATH=str(args.tambua_src.resolve()))
    if args.r_library:
        env["R_LIBS_USER"] = str(args.r_library.resolve())
    def cli(*command: str) -> None:
        subprocess.run([sys.executable, "-m", "tambua.cli", *command],
                       check=True, env=env)
    development_table = args.output / "fit_and_development_cells.csv"
    cli("build-table", "--manifest", str(args.output / "fit_and_development_manifest.csv"),
        "--output", str(development_table), "--compartments", "nuclei-cell")
    model = args.output / "fastpls_0.4_all23.rds"
    cli("train", "--table", str(development_table), "--model", str(model),
        "--report", str(args.output / "development_report.json"))
    test_table = args.output / "new_test_matched_cells.csv"
    cli("build-table", "--manifest", str(args.output / "new_test_manifest.csv"),
        "--output", str(test_table), "--compartments", "nuclei-cell")
    test_prediction = args.output / "new_test_matched_predictions.csv"
    cli("predict", "--table", str(test_table), "--model", str(model),
        "--output", str(test_prediction))
    all_table = args.output / "new_test_all_detected_cells.csv"
    cli("build-inference", "--manifest", str(args.output / "new_test_all_detected_manifest.csv"),
        "--output", str(all_table), "--compartments", "nuclei-cell")
    all_prediction = args.output / "new_test_all_detected_predictions.csv"
    cli("predict", "--table", str(all_table), "--model", str(model),
        "--output", str(all_prediction))
    development = json.loads((args.output / "development_report.json").read_text())
    if development["fastpls_version"] != "0.4" or len(development["feature_names"]) != 46 or \
            set(development["test_patients"]) & set(development["train_patients"]) or \
            set(development["test"]["classes"]) != set(CLASSES):
        raise ValueError("Wrong fastPLS version, panel or patient separation")
    inferred = rows(all_table)
    predicted = rows(all_prediction)
    if len(inferred) != len(predicted) or \
            {(r["sample_id"], r["label_id"]) for r in inferred} != \
            {(r["sample_id"], r["label_id"]) for r in predicted}:
        raise ValueError("All-detected test predictions differ from marker-table IDs")
    result = {"status": "complete", "device": "CPU",
              "method": "GitHub fastPLS 0.4 SIMPLS-LDA, full 23-channel nucleus plus cell panel",
              "model_sha256": sha(model), "qc_passing_expansion_images": len(expanded["records"]),
              "images_by_split": dict(split_counts),
              "development": development["test"],
              "new_test": matched_metrics(test_table, test_prediction,
                                          development["test"]["classes"]),
              "unseen_patient_test": matched_metrics(
                  test_table, test_prediction, development["test"]["classes"],
                  only_patients={row["patient_id"] for row in cases
                                 if row["split"] == "test" and
                                 not row["patient_seen_in_original_test"]}),
              "all_detected_new_test_cells_predicted": len(predicted),
              "publication_status": "preliminary_until_all_included_images_pass_documented_manual_visual_review",
              "limit": "All additional test patients are held out from fitting, but some were seen in prior organizer-test auditing. The unseen-patient subset excludes every patient in the original 32-image organizer test; its tissue and class coverage may be incomplete. Matched-cell metrics do not measure false-positive detection in incompletely annotated tissue. Every included image still requires documented visual RGB, nuclear and shadow review before website publication."}
    staged = args.output / "STUDY_COMPLETE.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(args.output / "STUDY_COMPLETE.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-qc", "old-quant", "new-quant", "tambua-src", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--r-library", type=Path)
    summary = run(parser.parse_args())
    print(json.dumps({key: summary[key] for key in
                      ("qc_passing_expansion_images", "images_by_split", "new_test")}, indent=2))
