#!/usr/bin/env python3
"""Freeze a patient-separated development fold inside MoNuSAC training.

Only organizer source labels and tissue names determine this split. The
independent organizer test patients are not inspected for model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


# Two training patients per tissue, selected before expanded GigaTIME inference.
# Each pair contains expert macrophage and neutrophil examples; the remaining
# training patients retain both rare types in every tissue.
DEV_PATIENTS = (
    "TCGA-B6-A0WZ", "TCGA-E9-A22G",  # breast
    "TCGA-SX-A7SR", "TCGA-G7-A8LD",  # kidney
    "TCGA-55-1594", "TCGA-L4-A4E5",  # lung
    "TCGA-G9-6499", "TCGA-EJ-5517",  # prostate
)
CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")


def freeze(candidate: Path, output: Path) -> dict:
    raw = candidate.read_bytes()
    source = json.loads(raw)
    records = source["records"]
    if len(records) != 213 or source["additional_samples"] != 93:
        raise ValueError("Expanded cohort differs from the frozen candidate")
    dev = set(DEV_PATIENTS)
    if len(dev) != 8:
        raise ValueError("Duplicate development patient")
    groups = {"fit": [], "development": [], "test": []}
    for row in records:
        split = row["split"]
        if split == "train":
            groups["development" if row["patient_id"] in dev else "fit"].append(row)
        elif split == "test":
            if row["patient_id"] in dev:
                raise ValueError("Development patient appears in organizer test")
            groups["test"].append(row)
        else:
            raise ValueError(f"Unexpected organizer split: {split}")
    ids = {name: {row["patient_id"] for row in cases} for name, cases in groups.items()}
    if any(ids[a] & ids[b] for a, b in (("fit", "development"),
                                         ("fit", "test"), ("development", "test"))):
        raise ValueError("Patient leakage between fit, development and test")

    def describe(rows: list[dict], require_rare_by_tissue: bool) -> dict:
        count = Counter()
        tissues = Counter()
        tissue_classes = {tissue: Counter() for tissue in
                          ("breast", "kidney", "lung", "prostate")}
        for row in rows:
            tissues[row["organ"]] += 1
            count.update(row["expert_types"])
            tissue_classes[row["organ"]].update(row["expert_types"])
        if set(tissues) != set(tissue_classes) or \
                (require_rare_by_tissue and any(
                    tissue_classes[t][name] == 0 for t in tissue_classes
                    for name in ("macrophage", "neutrophil"))):
            raise ValueError("A fold lacks an organ or rare expert type")
        return {"images": len(rows), "patients": len({r["patient_id"] for r in rows}),
                "images_by_tissue": dict(sorted(tissues.items())),
                "expert_regions_by_class": {name: count[name] for name in CLASSES},
                "rare_regions_by_tissue": {
                    tissue: {name: tissue_classes[tissue][name]
                             for name in ("macrophage", "neutrophil")}
                    for tissue in sorted(tissue_classes)}}

    if (len(groups["fit"]), len(groups["development"]), len(groups["test"])) != (115, 32, 66):
        raise ValueError("Unexpected patient-fold image counts")
    result = {
        "status": "source-label-only_development_split_frozen_before_expanded_inference",
        "expanded_candidate_sha256": hashlib.sha256(raw).hexdigest(),
        "rule": "Two organizer-training patients per tissue in development; all remaining organizer-training patients fit candidate models. Organizer-test patients are reserved for one final evaluation.",
        "development_patients": list(DEV_PATIENTS),
        "summary": {name: describe(rows, name != "test")
                    for name, rows in groups.items()},
        "patient_sets": {name: sorted(values) for name, values in ids.items()},
        "limit": "Expert source counts precede virtual-marker QC; only complete QC passes may enter any subsequent fit, development score or final test score.",
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = freeze(args.candidate, args.output)
    print(json.dumps(result["summary"], indent=2))
