"""Matched, detection-aware cell-type scoring on frozen annotated images.

Usage: python -m analysis.score_instances --manifest cases.csv --method Tambua
       --track end_to_end --output results/tambua.json

The manifest has sample_id,patient_id,reference_mask,reference_labels,
prediction_mask,prediction_labels. Label CSVs have label_id,cell_type.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def labels(path: str | Path) -> dict[int, str]:
    output: dict[int, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            identifier, cell_type = int(row["label_id"]), row["cell_type"]
            if identifier <= 0 or identifier in output or cell_type not in CLASSES:
                raise ValueError(f"Invalid or duplicate cell label in {path}: {identifier}, {cell_type}")
            output[identifier] = cell_type
    return output


def _counts(mask: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(mask, return_counts=True)
    return {int(i): int(n) for i, n in zip(ids, counts) if i > 0}


def match_instances(reference: np.ndarray, prediction: np.ndarray,
                    threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """One-to-one IoU matches; the strict >0.5 criterion prevents ambiguous ties."""
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Reference and prediction masks must be aligned two-dimensional arrays")
    if reference.dtype.kind not in "ui" or prediction.dtype.kind not in "ui":
        raise ValueError("Instance masks must have unsigned/integer labels")
    r_counts, p_counts = _counts(reference), _counts(prediction)
    both = (reference > 0) & (prediction > 0)
    pairs, overlaps = np.unique(np.column_stack((reference[both], prediction[both])),
                                axis=0, return_counts=True)
    candidates = []
    for (r_id, p_id), intersection in zip(pairs, overlaps):
        r_id, p_id = int(r_id), int(p_id)
        union = r_counts[r_id] + p_counts[p_id] - int(intersection)
        iou = int(intersection) / union
        if iou > threshold:
            candidates.append((r_id, p_id, float(iou)))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    used_r, used_p, matches = set(), set(), []
    for r_id, p_id, iou in candidates:
        if r_id not in used_r and p_id not in used_p:
            matches.append((r_id, p_id, iou))
            used_r.add(r_id)
            used_p.add(p_id)
    return matches


def count_case(reference: np.ndarray, prediction: np.ndarray,
               reference_types: dict[int, str], prediction_types: dict[int, str]) -> dict:
    reference_ids, prediction_ids = set(_counts(reference)), set(_counts(prediction))
    if reference_ids != set(reference_types) or prediction_ids != set(prediction_types):
        raise ValueError("Every mask instance needs exactly one typed label and no extra label rows")
    matches = match_instances(reference, prediction)
    correct = sum(reference_types[r] == prediction_types[p] for r, p, _ in matches)
    return dict(reference=len(reference_ids), prediction=len(prediction_ids),
                matched=len(matches), correct_type=int(correct),
                reference_by_type={name: sum(value == name for value in reference_types.values()) for name in CLASSES},
                prediction_by_type={name: sum(value == name for value in prediction_types.values()) for name in CLASSES},
                correct_by_type={name: sum(reference_types[r] == prediction_types[p] == name
                                           for r, p, _ in matches) for name in CLASSES},
                mean_matched_iou=float(np.mean([iou for _, _, iou in matches])) if matches else None)


def summarize(cases: list[dict]) -> dict:
    total = {name: sum(row[name] for row in cases) for name in
             ("reference", "prediction", "matched", "correct_type")}
    r_by = {name: sum(row["reference_by_type"][name] for row in cases) for name in CLASSES}
    p_by = {name: sum(row["prediction_by_type"][name] for row in cases) for name in CLASSES}
    tp_by = {name: sum(row["correct_by_type"][name] for row in cases) for name in CLASSES}
    detection_denom = total["reference"] + total["prediction"]
    f1_by = {name: (2 * tp_by[name] / (r_by[name] + p_by[name])
                     if r_by[name] + p_by[name] else None) for name in CLASSES}
    recall_by = {name: (tp_by[name] / r_by[name] if r_by[name] else None) for name in CLASSES}
    complete = all(r_by[name] > 0 for name in CLASSES)
    total.update(detection_f1=2 * total["matched"] / detection_denom if detection_denom else None,
                 matched_classification_accuracy=(total["correct_type"] / total["matched"]
                                                  if total["matched"] else None),
                 type_f1=f1_by, type_recall=recall_by,
                 type_macro_f1=(float(np.mean(list(f1_by.values()))) if complete else None),
                 type_balanced_recall=(float(np.mean(list(recall_by.values()))) if complete else None),
                 missing_reference_classes=[name for name in CLASSES if r_by[name] == 0],
                 reference_by_type=r_by, prediction_by_type=p_by)
    return total


def score_manifest(path: str | Path, method: str, track: str) -> dict:
    if track not in ("end_to_end", "expert_mask_classification"):
        raise ValueError("Unknown evaluation track")
    cases, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            sample, patient = entry["sample_id"], entry["patient_id"]
            if not sample or not patient or sample in seen:
                raise ValueError("Sample and patient IDs must be present and samples unique")
            seen.add(sample)
            reference = np.asarray(tifffile.imread(entry["reference_mask"]))
            prediction = np.asarray(tifffile.imread(entry["prediction_mask"]))
            if track == "expert_mask_classification" and not np.array_equal(reference, prediction):
                raise ValueError("Expert-mask classification must use identical reference instances")
            counts = count_case(reference, prediction,
                                labels(entry["reference_labels"]), labels(entry["prediction_labels"]))
            cases.append(dict(sample_id=sample, patient_id=patient, **counts))
    if not cases:
        raise ValueError("Empty evaluation manifest")
    by_patient = defaultdict(list)
    for case in cases:
        by_patient[case["patient_id"]].append(case)
    return dict(method=method, track=track, iou_match_threshold="strictly greater than 0.5",
                sample_ids=sorted(seen), patient_ids=sorted(by_patient),
                overall=summarize(cases),
                by_patient={patient: summarize(group) for patient, group in sorted(by_patient.items())},
                by_sample={case["sample_id"]: case for case in cases},
                caveat="Same frozen held-out patients and ontology required for cross-method comparison.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--track", choices=("end_to_end", "expert_mask_classification"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = score_manifest(args.manifest, args.method, args.track)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} for {len(report['sample_ids'])} samples / {len(report['patient_ids'])} patients")


if __name__ == "__main__":
    main()
