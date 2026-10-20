#!/usr/bin/env python3
"""CPU-only leave-one-patient-out audit of class balance on saved fit cells.

This is development evaluation. It never reads organizer-test labels and does
not alter the deployed classifier. Repeated rare rows are weights, not new
biological examples; patient-held-out recall is the criterion for comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")
STRATEGIES = {
    "original": None,
    "capped_500_rare_50": {"epithelial": 500, "lymphocyte": 500,
                           "macrophage": 50, "neutrophil": 50},
    "equal_250": {name: 250 for name in CLASSES},
}


def sample_training(rows: list[dict], strategy: str, seed: int) -> list[dict]:
    target = STRATEGIES[strategy]
    if target is None:
        return rows
    rng = random.Random(seed)
    selected = []
    for name in CLASSES:
        members = [row for row in rows if row["cell_type"] == name]
        if len(members) < 2:
            raise ValueError(f"Too few training examples of {name}")
        count = target[name]
        if len(members) >= count:
            selected.extend(rng.sample(members, count))
        else:
            # Include every unique rare cell, then repeat with replacement.
            selected.extend(members)
            selected.extend(rng.choices(members, k=count - len(members)))
    rng.shuffle(selected)
    return selected


def write_input(path: Path, rows: list[dict], names: list[str], *, labelled: bool) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names + (["label"] if labelled else []))
        for row in rows:
            writer.writerow([row[name] for name in names] +
                            ([row["cell_type"]] if labelled else []))


def confusion(rows: list[dict], predictions: list[str]) -> dict:
    counts = {name: Counter() for name in CLASSES}
    for row, predicted in zip(rows, predictions, strict=True):
        if predicted not in CLASSES:
            raise ValueError("Unknown predicted type")
        counts[row["cell_type"]][predicted] += 1
    support = {name: sum(counts[name].values()) for name in CLASSES}
    if any(not support[name] for name in CLASSES):
        raise ValueError("Held-out fit patients have no example of a class")
    recall = {name: counts[name][name] / support[name] for name in CLASSES}
    return {"support": support, "recall": recall,
            "balanced_accuracy": sum(recall.values()) / len(CLASSES),
            "accuracy": sum(counts[name][name] for name in CLASSES) / len(rows),
            "matrix": [[counts[actual][predicted] for predicted in CLASSES]
                       for actual in CLASSES], "classes": CLASSES}


def run(input_path: Path, bridge: Path, output: Path, *, include_morph: bool = False) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This audit requires CUDA_VISIBLE_DEVICES=-1")
    if output.exists():
        raise FileExistsError(output)
    with input_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = [name for name in reader.fieldnames or []
                 if name.startswith("nuclei_") or
                 (name.startswith("cell_") and name != "cell_type") or
                 (include_morph and name.startswith("morph_"))]
        rows = [row for row in reader if row["split"] == "train"]
    if len(names) != (56 if include_morph else 46) or len(rows) != 10049:
        raise ValueError("Saved complete-channel fit cohort changed")
    if include_morph and len([name for name in names if name.startswith("morph_")]) != 10:
        raise ValueError("Expected exactly ten H&E morphology features")
    patients = sorted({row["patient_id"] for row in rows})
    if len(patients) != 21:
        raise ValueError("Saved fit-patient set changed")
    results = {}
    for strategy_index, strategy in enumerate(STRATEGIES):
        holdout_predictions: dict[tuple[str, str, str], str] = {}
        fit_sizes = []
        for fold, patient in enumerate(patients):
            train = [row for row in rows if row["patient_id"] != patient]
            test = [row for row in rows if row["patient_id"] == patient]
            if not test or {row["patient_id"] for row in train} & {patient}:
                raise ValueError("Patient leakage or empty fold")
            selected = sample_training(train, strategy, 314159 + strategy_index * 1000 + fold)
            fit_sizes.append(len(selected))
            with tempfile.TemporaryDirectory(prefix="tambua-cpu-fastpls-") as tmp:
                folder = Path(tmp)
                training, testing = folder / "train.csv", folder / "test.csv"
                model, fit_log, predictions = (folder / "model.rds", folder / "fit.csv",
                                               folder / "predictions.csv")
                write_input(training, selected, names, labelled=True)
                write_input(testing, test, names, labelled=False)
                for mode, source, destination in (("train", training, fit_log),
                                                  ("predict", testing, predictions)):
                    subprocess.run(["Rscript", "--vanilla", str(bridge), mode,
                                    str(source), str(model), str(destination)],
                                   check=True, stdout=subprocess.DEVNULL)
                with predictions.open(newline="") as stream:
                    predicted = [item["predicted"] for item in csv.DictReader(stream)]
            if len(predicted) != len(test):
                raise ValueError("Fold prediction count changed")
            for row, value in zip(test, predicted, strict=True):
                key = (row["sample_id"], row["patient_id"], row["label_id"])
                if key in holdout_predictions:
                    raise ValueError("Duplicate held-out cell")
                holdout_predictions[key] = value
        ordered = [holdout_predictions[(row["sample_id"], row["patient_id"],
                                        row["label_id"])] for row in rows]
        results[strategy] = {**confusion(rows, ordered),
                             "fit_rows_per_fold_min": min(fit_sizes),
                             "fit_rows_per_fold_max": max(fit_sizes)}
    result = {"status": "complete", "device": "CPU",
              "scope": "21-patient leave-one-out development evaluation on 10,049 saved QC-pass matched fit cells only",
              "feature_set": "all-23-GigaTIME-markers-plus-10-H&E-morphology" if include_morph else "all-23-GigaTIME-markers",
              "feature_count": len(names),
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "patients": patients, "strategies": results,
              "limit": "Repeated rare rows are class weights, not independent examples. No organizer-test labels entered training or selection. This development audit cannot establish final four-class accuracy."}
    output.mkdir(parents=True)
    staged = output / "RARE_BALANCE_AUDIT.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(output / "RARE_BALANCE_AUDIT.json")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--include-morph", action="store_true",
                    help="Append ten label-free scanner-RGB H&E nuclear features")
    args = ap.parse_args()
    report = run(args.input, args.bridge, args.output, include_morph=args.include_morph)
    print(json.dumps({name: {key: value for key, value in score.items()
                              if key in ("balanced_accuracy", "accuracy", "support", "recall")}
                      for name, score in report["strategies"].items()}, indent=2))
