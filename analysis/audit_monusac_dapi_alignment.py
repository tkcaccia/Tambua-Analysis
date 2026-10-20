#!/usr/bin/env python3
"""Read-only check for a common DAPI-to-nuclear-mask translation.

Use only frozen training cases for diagnosis. A best shift is exploratory and
does not change QC, registration, held-out test labels, or classifier inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile
import zarr


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else float("nan")


def score_shifts(dapi: np.ndarray, labels: np.ndarray,
                 max_shift: int = 32, stride: int = 2) -> dict:
    if dapi.ndim != 2 or labels.ndim != 2 or min(dapi.shape) <= 4 * max_shift:
        raise ValueError("DAPI/mask must be sufficiently large 2D fields")
    # Match the half-pixel index mapping in gigatime_nuclear_qc.py exactly.
    yy = np.minimum(labels.shape[0]-1,
                    np.floor((np.arange(dapi.shape[0])+.5) *
                             labels.shape[0] / dapi.shape[0]).astype(int))
    xx = np.minimum(labels.shape[1]-1,
                    np.floor((np.arange(dapi.shape[1])+.5) *
                             labels.shape[1] / dapi.shape[1]).astype(int))
    binary = (labels[yy[:, None], xx[None, :]] > 0).astype(np.float32)
    target = gaussian_filter(binary, sigma=3)
    h, w = dapi.shape
    margin = max_shift
    source = dapi[margin:h-margin:stride, margin:w-margin:stride]

    def at(dy: int, dx: int) -> float:
        shifted = target[margin+dy:h-margin+dy:stride,
                         margin+dx:w-margin+dx:stride]
        return correlation(source, shifted)

    zero = at(0, 0)
    candidates = [(dy, dx, at(dy, dx))
                  for dy in range(-max_shift, max_shift+1, 4)
                  for dx in range(-max_shift, max_shift+1, 4)]
    best_coarse = max(candidates, key=lambda item: item[2])
    local = [(dy, dx, at(dy, dx))
             for dy in range(max(-max_shift, best_coarse[0]-4),
                             min(max_shift, best_coarse[0]+4)+1)
             for dx in range(max(-max_shift, best_coarse[1]-4),
                             min(max_shift, best_coarse[1]+4)+1)]
    best = max(local, key=lambda item: item[2])
    return {"zero_shift_pearson": zero, "best_shift_dy_dx_model_px":
            [best[0], best[1]], "best_shift_pearson": best[2],
            "improvement": best[2] - zero,
            "interpretation": "Mask shifted relative to DAPI; exploratory train-only search, not a QC correction"}


def run(prepared: Path, inference: Path, samples: list[str], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = {r["sample_id"]: r for r in
            json.loads((prepared / "PREP_COMPLETE.json").read_text())["records"]}
    if samples == ["ALL_COMPLETED_TRAIN"]:
        samples = sorted(p.parent.name for p in inference.glob("*/CASE_COMPLETE.json")
                         if rows[p.parent.name]["split"] == "train")
    records = []
    for sample in samples:
        row = rows[sample]
        if row["split"] != "train":
            raise ValueError(f"Held-out test case forbidden in shift search: {sample}")
        case = inference / sample
        completed = json.loads((case / "CASE_COMPLETE.json").read_text())
        if completed["sample_id"] != sample or \
                abs(float(completed["model_mpp"]) - .25) > 1e-6:
            raise ValueError(f"Case identity or model grid changed: {sample}")
        store = zarr.open(str(case / "gigatime_output" / "gigatime_probs.zarr"), mode="r")
        if store.attrs["gigatime"]["store_channels"][0] != "DAPI":
            raise ValueError(f"First channel is not DAPI: {sample}")
        dapi = np.asarray(store["0"][0], dtype=np.float32) / 65535.0
        mask = tifffile.imread(case / "stardist_labels.tif")
        result = score_shifts(dapi, mask)
        records.append({"sample_id": sample, "organ": row["organ"],
                        "frozen_qc_status": completed["nuclear_qc"]["status"],
                        **result})
    report = {"status": "diagnostic_only", "cases": len(records),
              "max_shift_model_px": 32, "model_mpp_um": .25,
              "records": records,
              "limit": "The same H&E supports both the StarDist mask and virtual DAPI. A train-only shift search cannot establish measured-marker validity. No QC status is changed."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.prepared, a.inference, a.samples, a.output)))
