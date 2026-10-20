#!/usr/bin/env python3
"""Compare saved ORION virtual DAPI with registered raw measured Hoechst.

Read-only, CPU-only. This tests whether a shift inferred from MoNuSAC training
images generalizes to independently measured fluorescence. It does not infer
new marker images or change any QC and is not a protein-correlation result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import shift as nd_shift
import tifffile
import zarr


def normalize(image: np.ndarray) -> np.ndarray:
    image = np.log1p(np.asarray(image, dtype=np.float32))
    lo, hi = np.quantile(image[::4, ::4], [.005, .995])
    if hi <= lo:
        raise ValueError("Flat spatial image")
    return np.clip((image-lo)/(hi-lo), 0, 1).astype(np.float32)


def spatial_corr(measured: np.ndarray, virtual: np.ndarray,
                 dy: float, dx: float) -> float:
    h, w = measured.shape
    translated = nd_shift(virtual, (dy, dx), order=1,
                          mode="constant", cval=0, prefilter=False)
    margin = 32
    a, b = measured[margin:-margin, margin:-margin], \
        translated[margin:-margin, margin:-margin]
    a, b = a-a.mean(), b-b.mean()
    denom = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    return float(np.sum(a*b)/denom) if denom else float("nan")


def run(patients: list[str], measured_root: Path,
        virtual_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if len(patients) != 10 or len(set(patients)) != 10:
        raise ValueError("Use exactly ten distinct reserved ORION patients")
    records = []
    for patient in sorted(patients):
        measured_dir = measured_root/patient
        source = json.loads((measured_dir/"MEASURED_PIXEL_SOURCE.json").read_text())
        case = virtual_root/patient
        saved = json.loads((case/"CASE_COMPLETE.json").read_text())
        if source["patient"] != patient or saved["patient"] != patient or \
                not saved["technical_qc_passed"] or \
                abs(source["mpp_x"]-.325)>1e-6 or \
                abs(source["mpp_y"]-.325)>1e-6:
            raise ValueError(f"Identity, QC, or physical scale differs: {patient}")
        measured = tifffile.imread(measured_dir/"Hoechst.tif")
        store = zarr.open(str(case/"gigatime"/"gigatime_probs.zarr"), mode="r")
        if store["0"].shape != (1, 2663, 2663) or \
                store.attrs["gigatime"]["store_channels"] != ["DAPI"]:
            raise ValueError(f"Unexpected saved virtual grid: {patient}")
        virtual = np.asarray(store["0"][0])
        if measured.shape != (2048, 2048):
            raise ValueError(f"Unexpected registered measured grid: {patient}")
        m = np.asarray(Image.fromarray(normalize(measured), mode="F").resize(
            (1024, 1024), Image.Resampling.BILINEAR))
        v = np.asarray(Image.fromarray(normalize(virtual), mode="F").resize(
            (1024, 1024), Image.Resampling.BILINEAR))
        scale = 1024/2663
        zero = spatial_corr(m, v, 0, 0)
        fixed = spatial_corr(m, v, -3*scale, 3*scale)
        # Nuisance registration can differ between H&E and measured IF; report
        # the local best only as a diagnostic and never use it for evaluation.
        candidates = [(dy, dx, spatial_corr(m, v, dy, dx))
                      for dy in range(-8, 9, 2) for dx in range(-8, 9, 2)]
        coarse = max(candidates, key=lambda item: item[2])
        refined = [(dy, dx, spatial_corr(m, v, dy, dx))
                   for dy in range(max(-8, coarse[0]-2), min(8, coarse[0]+2)+1)
                   for dx in range(max(-8, coarse[1]-2), min(8, coarse[1]+2)+1)]
        best = max(refined, key=lambda item: item[2])
        records.append({"patient": patient, "measured": "Hoechst",
                        "virtual": "DAPI", "zero_shift_pearson": zero,
                        "train_derived_fixed_shift_pearson": fixed,
                        "fixed_minus_zero": fixed-zero,
                        "exploratory_local_best_dy_dx_1024px": [best[0],best[1]],
                        "exploratory_local_best_pearson": best[2]})
    report = {"status": "read_only_diagnostic", "patients": len(records),
              "pixel_grid": [1024, 1024], "source_mpp_um": .325,
              "virtual_mpp_um": .25, "fixed_shift_model_px": [-3, 3],
              "records": records,
              "limit": "DAPI and Hoechst are related nuclear dyes, not identical channels. Local optimization is exploratory and can overfit; no registration or QC is changed. Raw spatial correlations need separate marker-specific validation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix+".tmp")
    tmp.write_text(json.dumps(report, indent=2)+"\n")
    tmp.replace(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patients", nargs="+", required=True)
    p.add_argument("--measured-root", type=Path, required=True)
    p.add_argument("--virtual-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    x = run(a.patients, a.measured_root, a.virtual_root, a.output)
    print(json.dumps({"patients": x["patients"],
                      "fixed_minus_zero": [r["fixed_minus_zero"] for r in x["records"]]}))
