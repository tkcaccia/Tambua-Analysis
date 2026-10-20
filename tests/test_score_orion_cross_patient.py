"""Check patient weighting and frozen reserved-cohort admission."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import rankdata


path = Path(__file__).resolve().parents[1] / "analysis" / "score_orion_cross_patient.py"
spec = importlib.util.spec_from_file_location("score_orion_cross_patient", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_equal_patient_weight_despite_unequal_cell_counts():
    first = np.arange(30, dtype=float)
    second = np.arange(90, dtype=float)
    values = np.stack([module.moments(first, first),
                       module.moments(second, second[::-1])])
    x = np.r_[rankdata(first) / 31, rankdata(second) / 91]
    y = np.r_[rankdata(first) / 31, rankdata(second[::-1]) / 91]
    weight = np.r_[np.full(30, 1 / 60), np.full(90, 1 / 180)]
    covariance = np.average((x - np.average(x, weights=weight)) *
                            (y - np.average(y, weights=weight)), weights=weight)
    variance_x = np.average((x - np.average(x, weights=weight)) ** 2, weights=weight)
    variance_y = np.average((y - np.average(y, weights=weight)) ** 2, weights=weight)
    expected = covariance / np.sqrt(variance_x * variance_y)
    assert module.correlation_from_moments(values) == pytest.approx(expected, abs=1e-12)
    assert expected > -0.05  # 90-cell anti-correlation did not dominate 30-cell positive case.
    assert module.correlation_from_moments(values[[0, 0]]) == pytest.approx(1)


def test_reserved_qc_admission_requires_ten_unique_patients():
    summary = {"patient_records": [
        {"patient": f"CRC{number:02d}", "split": "evaluation",
         "technical_qc_passed": True} for number in range(10)
    ] + [{"patient": "CRC99", "split": "development",
          "technical_qc_passed": True}]}
    assert len(module.patient_ids(summary)) == 10
    summary["patient_records"][0]["technical_qc_passed"] = False
    with pytest.raises(ValueError, match="ten distinct"):
        module.patient_ids(summary)
