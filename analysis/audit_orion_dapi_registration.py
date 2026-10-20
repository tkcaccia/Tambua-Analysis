#!/usr/bin/env python3
"""CPU-only diagnostic of possible measured/virtual ORION DAPI pixel shifts.

Candidate shifts are selected on the same patient being scored, so this is a
registration sensitivity audit and never an independent corrected score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.stats import spearmanr
import tifffile
import zarr


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    aa -= aa.mean()
    bb -= bb.mean()
    return float(np.dot(aa, bb) / np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))


def one(study: Path, measured_root: Path, patient: str, max_shift: int = 2) -> dict:
    source = json.loads((measured_root / patient / "MEASURED_PIXEL_SOURCE.json").read_text())
    prep = json.loads((study / "cpu-prep" / patient / "PROVENANCE.json").read_text())
    if any(source[key] != prep[key] for key in ("x", "y", "size_px", "mpp_x", "mpp_y")):
        raise ValueError("ORION physical registration provenance changed")
    case = study / "gpu-output" / patient
    if not json.loads((case / "CASE_COMPLETE.json").read_text())["technical_qc_passed"]:
        raise ValueError("Virtual DAPI has not passed technical QC")
    if json.loads((case / "gigatime" / "gigatime_seam_qc.json").read_text())["status"] != "pass":
        raise ValueError("Virtual DAPI seam QC failed")
    store = zarr.open(str(case / "gigatime" / "gigatime_probs.zarr"), mode="r")
    meta = store.attrs["gigatime"]
    if meta["store_channels"] != ["DAPI"] or abs(meta["effective_mpp"] - .25) > 1e-6 or \
            abs(meta["source_mpp"] - prep["mpp_x"]) > 1e-6:
        raise ValueError("Saved virtual image is not calibrated DAPI only")
    size = prep["size_px"]
    virtual = np.asarray(store["0"][0], dtype=np.float32)
    virtual = zoom(virtual, (size / virtual.shape[0], size / virtual.shape[1]), order=1)
    raw = tifffile.imread(measured_root / patient / "Hoechst.tif").astype(np.float32)
    nucleus = tifffile.imread(study / "cpu-prep" / patient / "nuclei_orion_ids.ome.tif") > 0
    if virtual.shape != raw.shape or raw.shape != nucleus.shape or raw.shape != (size, size):
        raise ValueError("Registered image shapes disagree")
    stride = 4
    v, m, mask = virtual[::stride, ::stride], raw[::stride, ::stride], nucleus[::stride, ::stride]
    margin = max_shift + 1
    ys = slice(margin, v.shape[0] - margin)
    xs = slice(margin, v.shape[1] - margin)
    target, nuclear = m[ys, xs], mask[ys, xs]
    scores = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            candidate = v[margin + dy:v.shape[0] - margin + dy,
                          margin + dx:v.shape[1] - margin + dx]
            scores.append((correlation(candidate, target), dy, dx))
    best = max(scores)
    def spearman_at(dy: int, dx: int) -> tuple[float, float]:
        candidate = v[margin + dy:v.shape[0] - margin + dy,
                      margin + dx:v.shape[1] - margin + dx]
        return (float(spearmanr(candidate.ravel(), target.ravel()).statistic),
                float(spearmanr(candidate[nuclear], target[nuclear]).statistic))
    original = spearman_at(0, 0)
    shifted = spearman_at(best[1], best[2])
    return {"patient": patient, "zero_shift_full_spearman": original[0],
            "zero_shift_nuclear_spearman": original[1],
            "best_shift_downsampled_pixels_yx": [best[1], best[2]],
            "best_shift_source_pixels_yx": [best[1] * stride, best[2] * stride],
            "best_shift_full_spearman": shifted[0],
            "best_shift_nuclear_spearman": shifted[1],
            "nuclear_pixels_sampled": int(nuclear.sum())}


def run(study: Path, measured: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Registration audit must hide all GPUs")
    if output.exists():
        raise FileExistsError(output)
    cohort = json.loads((measured / "COHORT_COMPLETE.json").read_text())
    records = cohort["patients"] if isinstance(cohort["patients"], list) else cohort["records"]
    patients = [row["patient"] for row in records]
    if len(patients) != 10 or len(set(patients)) != 10:
        raise ValueError("Expected ten distinct measured patients")
    rows = [one(study, measured, patient) for patient in patients]
    result = {"status": "complete", "device": "CPU", "patients": rows,
              "shift_search_source_pixels": [-8, 8],
              "shift_step_source_pixels": 4,
              "scope": "Sensitivity of registered raw Hoechst and virtual DAPI to small integer-grid shifts",
              "limit": "Each best shift is fitted to the same measured patient image. Its correlations are optimistic and cannot be published as independent GigaTIME accuracy or used to silently alter registered imagery."}
    output.mkdir(parents=True)
    (output / "REGISTRATION_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = run(args.study, args.measured, args.output)
    for row in report["patients"]:
        print(row)
