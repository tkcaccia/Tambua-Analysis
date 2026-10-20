"""Mask geometry and conservative labels for CPU-only detected-cell quantification."""
from pathlib import Path
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from quantify_detected_monusac_cells import (  # noqa: E402
    cell_proxy, load_inference_validation, sha256, validated_assignments)


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


def test_cpu_case_validation_requires_source_identity_and_pinned_model(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "he.ome.tif").write_bytes(b"scanner-rgb-source")
    (prepared / "provenance.json").write_text(json.dumps({"mpp": .2527}))
    parent = tmp_path / "bundle"
    case = parent / "sample"
    case.mkdir(parents=True)
    (parent / "BUNDLE_IDENTITY.json").write_text(json.dumps({
        "device": "CPU", "model_revision": "a6556030bdb8071857559e12b921620aa1bbac5b"}))
    row = {"sample_id": "sample", "source_rgb_sha256": sha256(prepared / "he.ome.tif"),
           "source_mpp": .2527, "model_mpp": .25,
           "channels": [f"marker-{index}" for index in range(23)]}
    (case / "CASE_COMPLETE.json").write_text(json.dumps(row))
    validation, path = load_inference_validation(case, prepared, "sample")
    assert validation == {"source_color_reference": "scanner_svs_rgb",
                          "effective_mpp": .25}
    assert path == case / "CASE_COMPLETE.json"
    (prepared / "he.ome.tif").write_bytes(b"different-source")
    with pytest.raises(ValueError, match="identity"):
        load_inference_validation(case, prepared, "sample")
