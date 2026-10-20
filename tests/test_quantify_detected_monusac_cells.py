"""Mask geometry and conservative labels for CPU-only detected-cell quantification."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from quantify_detected_monusac_cells import cell_proxy, validated_assignments  # noqa: E402


def test_cell_proxy_uses_physical_halo_without_changing_nuclei() -> None:
    nuclei = np.zeros((12, 12), dtype=np.uint16)
    nuclei[3, 3] = 1
    nuclei[3, 9] = 2
    cell = cell_proxy(nuclei, .5, halo_um=1.0)
    assert cell[3, 3] == 1 and cell[3, 9] == 2
    assert cell[3, 5] == 1  # exactly 1 µm from nucleus 1
    assert cell[3, 6] == 0  # outside both 1 µm halos
    assert cell[3, 7] == 2
    assert cell[0, 0] == 0


def test_assignments_reject_missing_or_duplicate_detected_ids() -> None:
    detected = np.array([[0, 1, 2]], dtype=np.uint16)
    pair = {"detected_id": 1, "expert_id": 8, "cell_type": "macrophage",
            "rule": "unique_contained_nucleus"}
    assert validated_assignments({"assignments": [pair]}, detected) == [pair]
    with pytest.raises(ValueError, match="not unique"):
        validated_assignments({"assignments": [pair, pair]}, detected)
    with pytest.raises(ValueError, match="not unique"):
        validated_assignments({"assignments": [{**pair, "detected_id": 3}]}, detected)
