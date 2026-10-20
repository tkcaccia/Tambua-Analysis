#!/usr/bin/env python3
"""CPU-only registered raw-pixel ORION comparison of ten frozen patient fields.

This is a spatial diagnostic, not a claim that virtual marker intensities
quantify measured protein expression in single cells.
"""
from __future__ import annotations

import argparse
import hashlib
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

PAIRS = {
    "DAPI": ("Hoechst", "nuclear-dye family"),
    "CD68": ("CD68", "exact"), "CD4": ("CD4", "exact"),
    "CD8": ("CD8a", "CD8 alpha target"), "CD20": ("CD20", "exact"),
    "PD-L1": ("PD-L1", "exact"), "CD3": ("CD3e", "CD3 epsilon complex"),
    "PD-1": ("PD-1", "exact"), "Ki67": ("Ki67", "exact"),
    "CK": ("Pan-CK", "cytokeratin family"),
}
PATIENTS = ("CRC04", "CRC06", "CRC08", "CRC10", "CRC12", "CRC17",
            "CRC19", "CRC21", "CRC23", "CRC25")


def sha_buffer(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def score_case(patient: str, study: Path, measured_root: Path,
               pilot_root: Path, nine_root: Path) -> list[dict]:
    source = study / "cpu-prep" / patient
    predicted = (pilot_root if patient == "CRC04" else nine_root) / patient
    finished = json.loads((predicted / "TECHNICAL_COMPLETE.json").read_text())
    if finished.get("patient") != patient or \
       finished.get("status") != "technical_complete_pending_visual_and_protein_validation":
        raise ValueError(f"Full-panel technical completion missing: {patient}")
    provenance = json.loads((source / "PROVENANCE.json").read_text())
    measured_source = json.loads((measured_root / patient / "MEASURED_PIXEL_SOURCE.json").read_text())
    if any(provenance[key] != measured_source[key] for key in
           ("x", "y", "size_px", "mpp_x", "mpp_y")) or \
       provenance["size_px"] != 2048 or provenance["mpp_x"] != .325:
        raise ValueError(f"Measured/H&E registration metadata changed: {patient}")
    with Image.open(source / "he.ome.tif") as handle:
        rgb = np.asarray(handle.convert("RGB"))[::4, ::4].astype(np.int16)
    tissue = (rgb.mean(axis=-1) < 240) & (np.ptp(rgb, axis=-1) > 12)
    store = zarr.open(str(predicted / "gigatime" / "gigatime_probs.zarr"), mode="r")
    data = store["0"]
    if data.shape[0] != 23 or data.dtype != np.uint16 or \
       tuple(store.attrs["gigatime"]["store_channels"]) != CHANNELS:
        raise ValueError(f"Invalid GigaTIME marker store: {patient}")
    measured_by_name = {item["marker"]: item for item in measured_source["channels"]}
    rows = []
    for marker, (measured_name, mapping) in PAIRS.items():
        image = tifffile.imread(measured_root / patient / f"{measured_name}.tif")
        if image.shape != (2048, 2048) or image.dtype != np.uint16 or \
           sha_buffer(image) != measured_by_name[measured_name]["sha256"]:
            raise ValueError(f"Measured native pixels changed: {patient}/{measured_name}")
        virtual = np.asarray(data[CHANNELS.index(marker)], dtype=np.float32)
        virtual = zoom(virtual, (2048 / virtual.shape[0], 2048 / virtual.shape[1]), order=1)
        if virtual.shape != (2048, 2048):
            raise ValueError("Virtual-to-source physical grid conversion failed")
        x = image[::4, ::4][tissue].astype(np.float32)
        y = virtual[::4, ::4][tissue]
        if len(x) < 10000 or np.ptp(x) == 0 or np.ptp(y) == 0:
            raise ValueError(f"Insufficient variable tissue pixels: {patient}/{marker}")
        rho = float(spearmanr(x, y).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"Nonfinite rank association: {patient}/{marker}")
        rows.append({"patient": patient, "virtual": marker, "measured": measured_name,
                     "mapping": mapping, "pixel_spearman_rho": rho,
                     "sampled_tissue_pixels": int(len(x)), "sampling_stride_source_pixels": 4})
    return rows


def run(study: Path, measured_root: Path, pilot_root: Path,
        nine_root: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Raw-marker scorer must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    complete = json.loads((nine_root / "NINE_PATIENTS_COMPLETE.json").read_text())
    if [x["patient"] for x in complete["cases"]] != list(PATIENTS[1:]):
        raise ValueError("Nine-patient technical bundle incomplete")
    rows = [row for patient in PATIENTS for row in
            score_case(patient, study, measured_root, pilot_root, nine_root)]
    summary = []
    for marker, (measured, mapping) in PAIRS.items():
        values = np.asarray([r["pixel_spearman_rho"] for r in rows if r["virtual"] == marker])
        summary.append({"virtual": marker, "measured": measured, "mapping": mapping,
                        "patients": len(values), "patient_median_rho": float(np.median(values)),
                        "positive_patients": int(np.sum(values > 0)),
                        "patient_min_rho": float(np.min(values)),
                        "patient_max_rho": float(np.max(values))})
    report = {"status": "raw_spatial_diagnostic_pending_manual_review",
              "patients": list(PATIENTS), "source_mpp_um": .325, "model_mpp_um": .25,
              "sampling_stride_source_pixels": 4, "per_patient": rows,
              "equal_patient_summary": summary,
              "limitation": "One registered 2048-pixel window per patient. Raw pixel rank agreement is distinct from cell-level protein agreement, phenotype prediction, and whole-slide validation. Biology and artifacts require visual review of each patient and all 23 channels."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--measured-root", type=Path, required=True)
    p.add_argument("--pilot-root", type=Path, required=True)
    p.add_argument("--nine-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.study, a.measured_root, a.pilot_root, a.nine_root, a.output)
    print(json.dumps({"status": result["status"], "patients": len(result["patients"]),
                      "summary": result["equal_patient_summary"]}, indent=2))
