#!/usr/bin/env python3
"""Compare raw ORION fluorescence images with registered GigaTIME channels.

All patients carry equal weight. Full-field and predeclared nuclear/cytoplasm
pixel masks are reported separately; no marker, compartment or patient is picked
after observing a favorable correlation. Visual display transforms are absent.
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


PAIRS = {
    "DAPI": ("Hoechst", "nuclear-dye family"),
    "CD68": ("CD68", "exact"),
    "CD4": ("CD4", "exact"),
    "CD8": ("CD8a", "same CD8 alpha target"),
    "CD20": ("CD20", "exact"),
    "PD-L1": ("PD-L1", "exact"),
    "CD3": ("CD3e", "same CD3 epsilon complex"),
    "PD-1": ("PD-1", "exact"),
    "Ki67": ("Ki67", "exact"),
    "CK": ("Pan-CK", "cytokeratin family"),
}
COMPARTMENTS = ("full_field", "registered_nuclei", "registered_cytoplasm")


def measured_patients(source: dict) -> list[str]:
    if source.get("status") != "complete" or source.get("device") != "CPU" or \
            source.get("patients") != 10 or source.get("measured_marker_images") != 100:
        raise ValueError("Measured spatial-image cohort is incomplete")
    patients = [item["patient"] for item in source["records"]]
    if len(patients) != 10 or len(set(patients)) != 10:
        raise ValueError("Expected ten distinct reserved QC-pass patients")
    return patients


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 100 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def score_patient(study: Path, measured_root: Path, virtual_root: Path,
                  patient: str, markers: list[str], stride: int) -> list[dict]:
    prep = json.loads((study / "cpu-prep" / patient / "PROVENANCE.json").read_text())
    measured = measured_root / patient
    source = json.loads((measured / "MEASURED_PIXEL_SOURCE.json").read_text())
    if source["patient"] != patient or any(source[k] != prep[k]
                                            for k in ("x", "y", "size_px", "mpp_x", "mpp_y")):
        raise ValueError(f"Measured pixels are not on the registered H&E grid: {patient}")
    source_channels = {row["marker"] for row in source["channels"]}
    if any(PAIRS[marker][0] not in source_channels for marker in markers):
        raise ValueError(f"Missing measured protein pixels: {patient}")
    case = virtual_root / patient
    complete = json.loads((case / "CASE_COMPLETE.json").read_text())
    seam = json.loads((case / "gigatime/gigatime_seam_qc.json").read_text())
    if not complete["technical_qc_passed"] or seam["status"] != "pass":
        raise ValueError(f"GigaTIME technical QC failed: {patient}")
    store = zarr.open(str(case / "gigatime/gigatime_probs.zarr"), mode="r")
    meta = dict(store.attrs)["gigatime"]
    names = meta["store_channels"]
    missing = sorted(set(markers) - set(names))
    if missing:
        raise ValueError(f"Saved GigaTIME spatial output for {patient} has only {names}; "
                         f"missing {missing}. All-marker inference is needed before protein pixel scoring")
    if len(set(names)) != len(names) or \
            abs(meta["source_mpp"] - prep["mpp_x"]) > 1e-6 or \
            abs(meta["effective_mpp"] - .25) > 1e-6:
        raise ValueError(f"GigaTIME channel identity or physical scale changed: {patient}")
    size = int(prep["size_px"])
    masks = {"full_field": np.ones((size, size), dtype=bool)}
    for key, filename in (("registered_nuclei", "nuclei_orion_ids.ome.tif"),
                          ("registered_cytoplasm", "cytoplasm_orion_ids.ome.tif")):
        array = tifffile.imread(study / "cpu-prep" / patient / filename)
        if array.shape != (size, size):
            raise ValueError(f"Compartment mask differs from H&E grid: {patient}/{key}")
        masks[key] = array > 0
    rows = []
    for marker in markers:
        target, relation = PAIRS[marker]
        virtual = np.asarray(store["0"][names.index(marker)], dtype=np.float32)
        if virtual.shape != tuple(meta["inference_shape_yx"]):
            raise ValueError(f"Native virtual marker dimensions changed: {patient}/{marker}")
        aligned = zoom(virtual, (size/virtual.shape[0], size/virtual.shape[1]), order=1)
        raw = tifffile.imread(measured / f"{target}.tif").astype(np.float32)
        if aligned.shape != raw.shape or raw.shape != (size, size):
            raise ValueError(f"Measured and virtual pixel grids differ: {patient}/{marker}")
        a, b = aligned[::stride, ::stride], raw[::stride, ::stride]
        for compartment in COMPARTMENTS:
            selected = masks[compartment][::stride, ::stride]
            rows.append({"patient": patient, "virtual_marker": marker,
                         "measured_marker": target, "relation": relation,
                         "compartment": compartment,
                         "pixels": int(selected.sum()),
                         "spearman_rho": safe_spearman(a[selected], b[selected]),
                         "source_mpp": prep["mpp_x"], "model_mpp": meta["effective_mpp"]})
    return rows


def run(study: Path, measured: Path, virtual: Path, markers: list[str], output: Path,
        stride: int = 4) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Spatial scoring must explicitly hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    if not markers or len(set(markers)) != len(markers) or any(name not in PAIRS for name in markers):
        raise ValueError("Marker list is empty, duplicated or not predeclared")
    if stride < 1:
        raise ValueError("Pixel stride must be positive")
    source = json.loads((measured / "COHORT_COMPLETE.json").read_text())
    patients = measured_patients(source)
    rows = [row for patient in patients for row in
            score_patient(study, measured, virtual, patient, markers, stride)]
    aggregates = []
    rng = np.random.default_rng(20261020)
    for marker in markers:
        for compartment in COMPARTMENTS:
            subset = [row for row in rows if row["virtual_marker"] == marker and
                      row["compartment"] == compartment]
            rho = [row["spearman_rho"] for row in subset]
            if len(subset) != 10 or any(value is None for value in rho):
                raise ValueError(f"Missing patient or undefined rank correlation: {marker}/{compartment}")
            values = np.asarray(rho, dtype=np.float64)
            boot = values[rng.integers(0, 10, size=(10000, 10))]
            aggregates.append({"virtual_marker": marker,
                               "measured_marker": PAIRS[marker][0],
                               "relation": PAIRS[marker][1],
                               "compartment": compartment, "patients": 10,
                               "equal_patient_median_spearman": float(np.median(values)),
                               "patient_bootstrap_95pct_ci": [float(x) for x in
                                   np.quantile(np.median(boot, axis=1), [.025, .975])]})
    result = {"status": "complete", "device": "CPU", "patients": 10,
              "pixel_stride": stride, "markers": markers,
              "scope": "Registered raw ORION fluorescence versus native spatial GigaTIME channels",
              "compartment_source": "Published registered ORION nuclear and cytoplasm masks; not H&E StarDist detections",
              "display_transform_used_for_scoring": False,
              "per_patient": rows, "equal_patient_aggregates": aggregates,
              "limit": "Ten previously analyzed QC-passing reserved patients, one window per patient. Full-field nuclear/background localization can inflate association relative to within-nucleus intensity ranks. Rank association is not protein concentration calibration or whole-slide validation."}
    output.mkdir(parents=True)
    staged = output / "SPATIAL_PIXEL_SCORE.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(output / "SPATIAL_PIXEL_SCORE.json")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--virtual", type=Path, required=True)
    ap.add_argument("--markers", nargs="+", default=list(PAIRS))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    report = run(args.study, args.measured, args.virtual, args.markers, args.output, args.stride)
    print(json.dumps({"patients": report["patients"], "aggregates": report["equal_patient_aggregates"]}))
