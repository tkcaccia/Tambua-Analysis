"""Check macrophage cell polygons map to one detected nucleus only."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from match_monusac_expert_to_detected import assign  # noqa: E402


def test_unique_contained_nucleus_recovers_whole_cell_label() -> None:
    expert = np.zeros((12, 12), dtype=np.uint16)
    detected = np.zeros_like(expert)
    expert[1:6, 1:6] = 1  # macrophage whole cell, 25 pixels
    detected[2:4, 2:4] = 1  # one nucleus, 4 pixels, IoU < 0.5
    expert[8:11, 8:11] = 2  # lymphocyte nucleus
    detected[8:11, 8:11] = 2
    pairs, details = assign(expert, detected, {1: "macrophage", 2: "lymphocyte"})
    by_class = {item["cell_type"]: item for item in pairs}
    assert by_class["macrophage"]["rule"] == "unique_contained_nucleus"
    assert by_class["lymphocyte"]["rule"] == "nuclear_iou"
    assert details["unmatched_expert_regions"] == 0


def test_multiple_nuclei_inside_one_cell_are_withheld() -> None:
    expert = np.zeros((12, 12), dtype=np.uint16)
    detected = np.zeros_like(expert)
    expert[1:10, 1:10] = 1
    detected[2:4, 2:4] = 1
    detected[7:9, 7:9] = 2
    pairs, details = assign(expert, detected, {1: "macrophage"})
    assert pairs == []
    assert details["ambiguous_macrophage_regions"] == 1
