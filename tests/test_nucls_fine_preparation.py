"""Protect whole-instance exclusion in the NuCLS polyline preparation."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from prepare_nucls_fine_cell_cohort import prepare_one


def _source(rows):
    image = np.full((32, 32, 3), (190, 135, 170), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=("", "type", "main_classification",
                                               "raw_classification", "coords_x", "coords_y"))
    writer.writeheader()
    for index, (name, xs, ys) in enumerate(rows, 1):
        writer.writerow({"": str(index), "type": "polyline",
                         "main_classification": name, "raw_classification": name,
                         "coords_x": ",".join(map(str, xs)),
                         "coords_y": ",".join(map(str, ys))})
    return {"fov_id": "synthetic", "image": {"bytes": stream.getvalue()},
            "annotations_csv": table.getvalue()}


def _run(tmp_path, rows):
    planned = {"fov_id": "synthetic", "patient_id": "patient-a", "split": "train",
               "height": 32, "width": 32, "target_polyline_counts": {}}
    result = prepare_one(_source(rows), planned, tmp_path)
    folder = tmp_path / "synthetic"
    labels = tifffile.imread(folder / "expert_target_nuclei.ome.tif")
    excluded = tifffile.imread(folder / "excluded_polyline_regions.ome.tif")
    assert not np.any((labels > 0) & (excluded > 0))
    return result, labels, excluded


def test_excluded_polygon_removes_entire_overlapping_target(tmp_path):
    rows = [
        ("macrophage", [2, 10, 10, 2], [2, 2, 10, 10]),
        ("other_stromal", [8, 13, 13, 8], [8, 8, 13, 13]),
        ("lymphocyte", [20, 26, 26, 20], [20, 20, 26, 26]),
    ]
    result, labels, excluded = _run(tmp_path, rows)
    assert result["target_polyline_counts_after_geometry_qc"] == {"lymphocyte": 1}
    assert labels[4, 4] == 0 and excluded[4, 4] == 1
    assert labels[23, 23] > 0


def test_overlapping_target_polygons_are_both_excluded(tmp_path):
    rows = [
        ("macrophage", [2, 10, 10, 2], [2, 2, 10, 10]),
        ("lymphocyte", [8, 14, 14, 8], [8, 8, 14, 14]),
    ]
    result, labels, excluded = _run(tmp_path, rows)
    assert result["typed_instance_count"] == 0
    assert not np.any(labels)
    assert excluded[4, 4] == 1 and excluded[12, 12] == 1
