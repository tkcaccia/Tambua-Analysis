#!/usr/bin/env python3
"""CPU-only local-shift sensitivity of registered measured/virtual ORION maps.

The maximum over post-hoc shifts is an optimistic diagnostic, not a valid
prediction score. It tests whether a small rigid offset could explain a low
zero-shift correlation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.stats import spearmanr
import tifffile
import zarr

from validate_orion_nine_raw_case import CHANNELS

PAIRS = {"DAPI": "Hoechst", "CD68": "CD68", "CK": "Pan-CK", "Ki67": "Ki67"}


def shifted_rho(measured: np.ndarray, virtual: np.ndarray,
                tissue: np.ndarray, dy: int, dx: int) -> float:
    h, w = tissue.shape
    ya, yb = max(0, dy), min(h, h + dy)
    xa, xb = max(0, dx), min(w, w + dx)
    yy, xx = ya-dy, xa-dx
    m = tissue[ya:yb, xa:xb] & tissue[yy:yy+yb-ya, xx:xx+xb-xa]
    a = measured[ya:yb, xa:xb][m]
    b = virtual[yy:yy+yb-ya, xx:xx+xb-xa][m]
    if len(a) < 10000:
        raise ValueError("Too few paired tissue pixels")
    return float(spearmanr(a, b).statistic)


def run(patient: str, source: Path, measured_root: Path,
        predicted_root: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Shift diagnostic is CPU-only")
    with Image.open(source / patient / "he.ome.tif") as handle:
        rgb = np.asarray(handle.convert("RGB"))[::8, ::8].astype(np.int16)
    tissue = (rgb.mean(-1) < 240) & (np.ptp(rgb, axis=-1) > 12)
    store = zarr.open(str(predicted_root / patient / "gigatime" /
                          "gigatime_probs.zarr"), mode="r")["0"]
    rows = []
    for marker, measured_name in PAIRS.items():
        observed = tifffile.imread(measured_root / patient / f"{measured_name}.tif")[::8, ::8]
        raw = np.asarray(store[CHANNELS.index(marker)], dtype=np.float32)
        predicted = zoom(raw, (2048/raw.shape[0], 2048/raw.shape[1]), order=1)[::8, ::8]
        if observed.shape != tissue.shape or predicted.shape != tissue.shape:
            raise ValueError("Registered diagnostic geometry changed")
        scores = [(dy, dx, shifted_rho(observed, predicted, tissue, dy, dx))
                  for dy in range(-2, 3) for dx in range(-2, 3)]
        best = max(scores, key=lambda item: item[2])
        baseline = next(rho for dy, dx, rho in scores if dy == dx == 0)
        rows.append({"marker": marker, "zero_shift_rho_stride8": baseline,
                     "optimistic_best_rho_within_16_source_pixels": best[2],
                     "best_shift_source_pixels_yx": [best[0]*8, best[1]*8],
                     "max_shift_source_pixels": 16})
    return {"patient": patient, "rows": rows,
            "interpretation": "Post-hoc best shift is optimistic and not an accuracy estimate. It only tests whether a small rigid misregistration can account for the low zero-shift association."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--measured", type=Path, required=True)
    p.add_argument("--predicted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = run(a.patient, a.source, a.measured, a.predicted)
    if a.output.exists():
        raise FileExistsError(a.output)
    temp = a.output.with_suffix(a.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2)+"\n")
    temp.replace(a.output)
    print(json.dumps(report))
