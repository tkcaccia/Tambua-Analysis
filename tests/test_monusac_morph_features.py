"""Physical-scale and channel-order checks for the CPU morphology audit."""

import json

import numpy as np
from PIL import Image
import pytest
import tifffile

from analysis.add_monusac_morph_features import image_features


def test_single_nucleus_uses_native_spacing_and_scanner_rgb(tmp_path):
    rgb = np.full((3, 3, 3), [128, 64, 32], dtype=np.uint8)
    labels = np.zeros((3, 3), dtype=np.uint16)
    labels[1, 1] = 1
    image_path = tmp_path / "he.ome.tif"
    mask_path = tmp_path / "stardist_labels.tif"
    provenance_path = tmp_path / "provenance.json"
    Image.fromarray(rgb).save(image_path)
    tifffile.imwrite(mask_path, labels)
    provenance_path.write_text(json.dumps({"mpp": 0.25,
                                          "source_color_reference": "scanner_svs_rgb"}))

    cell = image_features(image_path, mask_path, provenance_path)[1]
    assert cell["morph_area_um2"] == pytest.approx(0.0625)
    assert cell["morph_perimeter_um"] == pytest.approx(1.0)
    assert cell["morph_od_red_mean"] < cell["morph_od_green_mean"]
    assert cell["morph_od_green_mean"] < cell["morph_od_blue_mean"]

    provenance_path.write_text(json.dumps({"mpp": 0.19448,
                                          "source_color_reference": "scanner_svs_rgb"}))
    finer = image_features(image_path, mask_path, provenance_path)[1]
    assert finer["morph_area_um2"] == pytest.approx(0.19448**2)

    provenance_path.write_text(json.dumps({"mpp": 0.25,
                                          "source_color_reference": "unverified"}))
    with pytest.raises(ValueError, match="scanner RGB"):
        image_features(image_path, mask_path, provenance_path)
