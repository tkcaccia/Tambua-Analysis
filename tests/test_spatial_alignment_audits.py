"""Synthetic checks for the read-only DAPI alignment diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import shift


ROOT = Path(__file__).resolve().parents[1] / "analysis"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_monusac_search_recovers_known_small_translation() -> None:
    module = load("audit_monusac_dapi_alignment")
    mask = np.zeros((260, 260), dtype=np.uint8)
    mask[90:110, 100:120] = 1
    mask[150:170, 175:195] = 1
    dapi = shift(mask.astype(float), shift=(7, -5), order=1)
    result = module.score_shifts(dapi, mask, max_shift=16, stride=2)
    dy, dx = result["best_shift_dy_dx_model_px"]
    assert abs(dy + 7) <= 1
    assert abs(dx - 5) <= 1
    assert result["improvement"] > 0.3


def test_orion_fixed_shift_degrades_already_aligned_images() -> None:
    module = load("audit_orion_dapi_pixel_alignment")
    yy, xx = np.mgrid[:180, :180]
    reference = (np.exp(-((yy-46)**2+(xx-53)**2)/100)
                 + np.exp(-((yy-122)**2+(xx-130)**2)/130)).astype(np.float32)
    assert module.spatial_corr(reference, reference, 0, 0) > .999
    assert module.spatial_corr(reference, reference, -3, 3) < .95
