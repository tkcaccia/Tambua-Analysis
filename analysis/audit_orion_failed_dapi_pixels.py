#!/usr/bin/env python3
"""Compare saved virtual DAPI with measured ORION Hoechst, including QC failures.

Exploratory QC audit only: a failed case is not promoted to a validation case.
This reads existing GigaTIME output and registered public fluorescence pixels;
it never runs GigaTIME or adjusts a threshold using the images it scores.
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


def score(study: Path, measured_root: Path, patient: str) -> dict:
    prep = json.loads((study / "cpu-prep" / patient / "PROVENANCE.json").read_text())
    source = json.loads((measured_root / patient / "MEASURED_PIXEL_SOURCE.json").read_text())
    if source["patient"] != patient or any(source[k] != prep[k] for k in ("x", "y", "size_px")) or \
            any(abs(source[k] - prep[k]) > 1e-6 for k in ("mpp_x", "mpp_y")):
        raise ValueError(f"Measured and H&E windows are not registered: {patient}")
    case = study / "gpu-output" / patient
    complete = json.loads((case / "CASE_COMPLETE.json").read_text())
    qc = json.loads((case / "gigatime/gigatime_gigatime_nuclear_qc.json").read_text())
    seam = json.loads((case / "gigatime/gigatime_seam_qc.json").read_text())
    store = zarr.open(str(case / "gigatime/gigatime_probs.zarr"), mode="r")
    meta = dict(store.attrs)["gigatime"]
    if meta["store_channels"] != ["DAPI"] or abs(meta["effective_mpp"] - .25) > 1e-6 or \
            abs(meta["source_mpp"] - prep["mpp_x"]) > 1e-6 or seam["status"] != "pass":
        raise ValueError(f"Virtual DAPI identity, scale or seam QC differs: {patient}")
    virtual = np.asarray(store["0"][0], dtype=np.float32)
    measured = tifffile.imread(measured_root / patient / "Hoechst.tif").astype(np.float32)
    size = prep["size_px"]
    if measured.shape != (size, size) or virtual.shape != tuple(meta["inference_shape_yx"]):
        raise ValueError(f"Registered pixel shapes differ: {patient}")
    aligned = zoom(virtual, (size / virtual.shape[0], size / virtual.shape[1]), order=1)
    if aligned.shape != measured.shape:
        raise ValueError(f"Model-grid resampling differs: {patient}")
    rho = float(spearmanr(aligned[::4, ::4].ravel(), measured[::4, ::4].ravel()).statistic)
    if not np.isfinite(rho):
        raise ValueError(f"Measured/virtual pixel correlation undefined: {patient}")
    return {"patient": patient, "source_mpp": prep["mpp_x"],
            "model_mpp": meta["effective_mpp"], "source_xy": [prep["x"], prep["y"]],
            "pixels_sampled": int(aligned[::4, ::4].size),
            "measured_virtual_pixel_spearman": rho,
            "native_qc_status": qc["status"],
            "native_nuclear_auc": qc["auc"],
            "native_boundary_contrast": qc["nuclear_boundary_contrast"],
            "native_spatial_shadow_status": qc["spatial_shadow_status"],
            "original_case_technical_qc_passed": complete["technical_qc_passed"],
            "measured_source_url": source["source_url"]}


def run(study: Path, measured_root: Path, patients: list[str], output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This diagnostic must explicitly hide all GPUs")
    if len(set(patients)) != len(patients) or output.exists():
        raise ValueError("Duplicate patients or existing output")
    rows = [score(study, measured_root, patient) for patient in patients]
    by_status = {}
    for status in ("pass", "fail"):
        values = [row["measured_virtual_pixel_spearman"] for row in rows
                  if row["native_qc_status"] == status]
        by_status[status] = {"patients": len(values),
                             "equal_patient_median_spearman":
                             (float(np.median(values)) if values else None)}
    result = {"status": "complete", "device": "CPU",
              "scope": "Exploratory measured-pixel check of existing ORION GigaTIME DAPI QC failures; no QC threshold changed",
              "patients": len(rows), "by_original_qc_status": by_status,
              "pixel_stride": 4,
              "metric": "Within-patient raw registered Hoechst versus native virtual DAPI Spearman; equal-patient medians by original QC status",
              "records": rows,
              "limit": "Using measured data to inspect a QC gate makes this a development audit, not independent validation. Do not admit failing cases solely because their correlation is high."}
    output.mkdir(parents=True)
    staged = output / "QC_PIXEL_AUDIT.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(output / "QC_PIXEL_AUDIT.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study, args.measured, args.patients, args.output)
    print(json.dumps({k: result[k] for k in ("patients", "by_original_qc_status")}))
