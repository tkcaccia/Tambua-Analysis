from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest
import tifffile

from analysis.qc_dapi import inspect


def test_dapi_qc_respects_mpp_and_detects_nuclear_alignment(tmp_path):
    he = np.empty((128, 128, 3), dtype=np.uint8)
    he[:] = (235, 178, 195)
    nucleus = np.zeros((128, 128), dtype=bool)
    for cy, cx in ((24, 25), (31, 91), (77, 45), (101, 101)):
        nucleus[cy-6:cy+6, cx-6:cx+6] = True
    he[nucleus] = (72, 42, 99)
    tifffile.imwrite(tmp_path / "he.ome.tif", he, ome=True, photometric="rgb",
                     metadata={"axes": "YXS", "PhysicalSizeX": .325,
                               "PhysicalSizeY": .325, "PhysicalSizeXUnit": "µm",
                               "PhysicalSizeYUnit": "µm"})
    tifffile.imwrite(tmp_path / "cells.tif", np.ones((128, 128), dtype=np.uint16))
    target = (167, 167)
    bright = np.asarray(Image.fromarray((nucleus * 255).astype(np.uint8)).resize(target,
                                Image.Resampling.NEAREST))
    Image.fromarray(bright).save(tmp_path / "aligned.png")
    Image.fromarray(np.roll(bright, 45, axis=1)).save(tmp_path / "shifted.png")
    manifest = dict(image_shape_yx=list(target), metadata=dict(source_mpp=.325,
                    effective_mpp=.25, original_shape_yx=[128, 128],
                    inference_shape_yx=list(target)))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    args = (tmp_path / "he.ome.tif", tmp_path / "cells.tif", tmp_path / "manifest.json")
    aligned = inspect(args[0], tmp_path / "aligned.png", args[1], args[2], tmp_path / "aligned")
    shifted = inspect(args[0], tmp_path / "shifted.png", args[1], args[2], tmp_path / "shifted")
    assert aligned["overlap_enrichment"] > shifted["overlap_enrichment"]
    assert aligned["review_status"] == "manual visual review required"
    assert (tmp_path / "aligned" / "he_dapi_overlay.png").exists()
    manifest["metadata"]["effective_mpp"] = .5
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="resolution contract"):
        inspect(args[0], tmp_path / "aligned.png", args[1], args[2], tmp_path / "invalid")
