"""Protect patient independence before expanded virtual-marker evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/monusac-corrected"


def test_frozen_development_fold_uses_only_organizer_training_patients() -> None:
    candidate = ASSETS / "candidate-expansion-256.json"
    fold = json.loads((ASSETS / "candidate-development-fold-256.json").read_text())
    rows = json.loads(candidate.read_text())["records"]
    assert fold["expanded_candidate_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    train_patients = {r["patient_id"] for r in rows if r["split"] == "train"}
    test_patients = {r["patient_id"] for r in rows if r["split"] == "test"}
    dev = set(fold["development_patients"])
    assert len(dev) == 8
    assert dev <= train_patients
    assert dev.isdisjoint(test_patients)
    assert set(fold["patient_sets"]["fit"]) == train_patients - dev
    assert set(fold["patient_sets"]["development"]) == dev
    assert set(fold["patient_sets"]["test"]) == test_patients
    assert [fold["summary"][group]["images"] for group in
            ("fit", "development", "test")] == [115, 32, 66]
    for group in ("fit", "development"):
        assert all(value > 0 for counts in fold["summary"][group]["rare_regions_by_tissue"].values()
                   for value in counts.values())
