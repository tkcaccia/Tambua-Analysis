import copy
import argparse
import json
from pathlib import Path

import pytest

from analysis.score_expanded_detected_cells import plan, run


ROOT = Path(__file__).resolve().parents[1]


def test_expanded_score_requires_explicit_exploratory_label():
    with pytest.raises(RuntimeError, match="no measured-marker validation"):
        run(argparse.Namespace(exploratory=False))


def test_expanded_plan_keeps_old_test_retrospective_and_blocks_patient_leakage():
    old = json.loads((ROOT / "docs/assets/monusac-corrected/stardist-nuclear-qc-summary.json").read_text())
    new = {"status": "complete", "device": "CPU", "source_images": 93,
           "qc_passing_images": 2,
           "records": [
               {"sample_id": "new-fit", "patient_id": "new-fit-patient",
                "organ": "breast", "split": "train"},
               {"sample_id": "new-test", "patient_id": "new-test-patient",
                "organ": "breast", "split": "test"}]}
    cases = plan(old, new)
    assert len(cases) == 49
    assert sum(row["split"] == "historical_test" for row in cases) == 10
    assert sum(row["split"] == "test" for row in cases) == 1
    leaked = copy.deepcopy(new)
    leaked["records"][1]["patient_id"] = "new-fit-patient"
    with pytest.raises(ValueError, match="Patient leakage"):
        plan(old, leaked)
    previously_tested = copy.deepcopy(new)
    previously_tested["records"][1]["patient_id"] = next(
        row["patient_id"] for row in old["records"] if row["split"] == "test")
    additional = plan(old, previously_tested)
    assert next(row for row in additional if row["sample_id"] == "new-test")[
        "patient_seen_in_original_test"] is True
