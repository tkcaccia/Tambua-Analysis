from __future__ import annotations

import csv

import numpy as np
import pytest
import tifffile

from analysis.score_instances import count_case, match_instances, score_manifest, summarize


def test_instance_matching_counts_wrong_type_and_unmatched_cells():
    reference = np.array([[1, 1, 0, 2, 2], [1, 1, 0, 2, 2], [0, 0, 0, 0, 0]], dtype=np.uint32)
    prediction = np.array([[7, 7, 0, 8, 8], [7, 7, 0, 8, 8], [0, 0, 0, 9, 0]], dtype=np.uint32)
    matched = match_instances(reference, prediction)
    assert [(r, p) for r, p, _ in matched] == [(1, 7), (2, 8)]
    result = count_case(reference, prediction,
                        {1: "epithelial", 2: "lymphocyte"},
                        {7: "epithelial", 8: "macrophage", 9: "lymphocyte"})
    assert result["correct_type"] == 1
    aggregate = summarize([result])
    assert aggregate["detection_f1"] == pytest.approx(4 / 5)
    assert aggregate["type_f1"]["epithelial"] == 1
    assert aggregate["type_f1"]["lymphocyte"] == 0
    assert aggregate["type_macro_f1"] is None
    assert set(aggregate["missing_reference_classes"]) == {"macrophage", "neutrophil"}


def test_expert_mask_track_rejects_changed_instances(tmp_path):
    reference = np.array([[0, 1], [0, 1]], dtype=np.uint16)
    prediction = np.array([[0, 1], [0, 0]], dtype=np.uint16)
    tifffile.imwrite(tmp_path / "reference.tif", reference)
    tifffile.imwrite(tmp_path / "prediction.tif", prediction)
    for name in ("reference", "prediction"):
        with (tmp_path / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("label_id", "cell_type"))
            writer.writerow((1, "epithelial"))
    with (tmp_path / "cases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "patient_id", "reference_mask", "reference_labels",
                         "prediction_mask", "prediction_labels"))
        writer.writerow(("s1", "p1", tmp_path / "reference.tif", tmp_path / "reference.csv",
                         tmp_path / "prediction.tif", tmp_path / "prediction.csv"))
    with pytest.raises(ValueError, match="identical"):
        score_manifest(tmp_path / "cases.csv", "Tambua", "expert_mask_classification")


def test_mask_shape_and_missing_label_rejected():
    reference = np.array([[1, 1]], dtype=np.uint16)
    prediction = np.array([[2, 2]], dtype=np.uint16)
    with pytest.raises(ValueError, match="exactly one"):
        count_case(reference, prediction, {}, {2: "epithelial"})
    with pytest.raises(ValueError, match="aligned"):
        match_instances(reference, np.array([[2], [2]], dtype=np.uint16))
