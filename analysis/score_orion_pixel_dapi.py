#!/usr/bin/env python3
"""Score registered raw Hoechst versus virtual DAPI across ORION patients.

This is a pixel-image comparison, unlike the published cell-table correlation.
The primary aggregate gives every patient equal weight; the bootstrap resamples
patients rather than treating millions of pixels as independent specimens.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.stats import spearmanr
import tifffile
import zarr


def one_patient(study: Path, measured_root: Path, patient: str,
                stride: int = 4) -> dict:
    prep = json.loads((study / "cpu-prep" / patient / "PROVENANCE.json").read_text())
    measured_dir = measured_root / patient
    source = json.loads((measured_dir / "MEASURED_PIXEL_SOURCE.json").read_text())
    if source["patient"] != patient or any(source[key] != prep[key]
                                            for key in ("x", "y", "size_px")):
        raise ValueError(f"Measured window and registered H&E mismatch: {patient}")
    if abs(source["mpp_x"] - prep["mpp_x"]) > 1e-6 or \
            abs(source["mpp_y"] - prep["mpp_y"]) > 1e-6:
        raise ValueError(f"Measured physical scale mismatch: {patient}")
    case = study / "gpu-output" / patient
    complete = json.loads((case / "CASE_COMPLETE.json").read_text())
    seam = json.loads((case / "gigatime/gigatime_seam_qc.json").read_text())
    if not complete["technical_qc_passed"] or seam["status"] != "pass":
        raise ValueError(f"Virtual image is not QC-accepted: {patient}")
    store = zarr.open(str(case / "gigatime/gigatime_probs.zarr"), mode="r")
    attrs = dict(store.attrs)["gigatime"]
    if attrs["store_channels"] != ["DAPI"] or abs(attrs["source_mpp"] - prep["mpp_x"]) > 1e-6 or \
            abs(attrs["effective_mpp"] - 0.25) > 1e-6:
        raise ValueError(f"Virtual DAPI channel or resolution mismatch: {patient}")
    virtual = np.asarray(store["0"][0], dtype=np.float32)
    measured = tifffile.imread(measured_dir / "Hoechst.tif").astype(np.float32)
    size = prep["size_px"]
    if measured.shape != (size, size) or virtual.shape != tuple(attrs["inference_shape_yx"]):
        raise ValueError(f"Measured or virtual image shape changed: {patient}")
    aligned = zoom(virtual, (size / virtual.shape[0], size / virtual.shape[1]), order=1)
    if aligned.shape != measured.shape:
        raise ValueError("Physical-grid interpolation changed output shape")
    a, b = aligned[::stride, ::stride].ravel(), measured[::stride, ::stride].ravel()
    rho = float(spearmanr(a, b).statistic)
    if not np.isfinite(rho):
        raise ValueError(f"Pixel correlation undefined: {patient}")
    border = max(1, size // 16)
    def edge_ratio(image: np.ndarray) -> float:
        edge = np.concatenate((image[:border].ravel(), image[-border:].ravel(),
                               image[border:-border, :border].ravel(),
                               image[border:-border, -border:].ravel()))
        return float(edge.mean() / image[border:-border, border:-border].mean())
    return {"patient": patient, "pixels_sampled": len(a), "spearman_rho": rho,
            "source_mpp": prep["mpp_x"], "virtual_mpp": attrs["effective_mpp"],
            "measured_border_core_ratio": edge_ratio(measured),
            "virtual_border_core_ratio": edge_ratio(aligned),
            "registered_window_xy": [prep["x"], prep["y"]]}


def run(study: Path, measured_root: Path, output: Path) -> dict:
    completed = json.loads((measured_root / "COHORT_COMPLETE.json").read_text())
    patients = [item["patient"] for item in completed["patients"]]
    if len(patients) != 10 or len(set(patients)) != 10:
        raise ValueError("Expected ten distinct reserved QC-passing patients")
    rows = [one_patient(study, measured_root, patient) for patient in patients]
    values = np.asarray([r["spearman_rho"] for r in rows])
    rng = np.random.default_rng(20261020)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))]
    result = {"patients": len(rows), "pixel_stride": 4,
              "measure": "same-section registered raw Hoechst pixels versus native GigaTIME virtual DAPI, bilinear-resampled to source physical grid",
              "equal_patient_median_spearman": float(np.median(values)),
              "equal_patient_median_95pct_bootstrap_ci": [float(v) for v in
                  np.quantile(np.median(samples, axis=1), [.025, .975])],
              "equal_patient_mean_spearman": float(values.mean()),
              "minimum_patient_spearman": float(values.min()),
              "maximum_patient_spearman": float(values.max()),
              "limit": "One 2048-pixel window per QC-passing patient; registered same-section comparison is not whole-WSI or calibrated-protein validation.",
              "per_patient": rows}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "per_patient.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.study, args.measured, args.output)
    print(json.dumps({key: result[key] for key in ("patients", "equal_patient_median_spearman",
                                                    "equal_patient_median_95pct_bootstrap_ci")}))
