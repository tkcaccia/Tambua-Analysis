#!/usr/bin/env python3
"""Read-only diagnostic of a train-derived fixed DAPI alignment shift.

The shift is estimated elsewhere on completed training fields. This script
applies the same frozen translation to every completed field, including held-
out test images, without selecting a better test-specific shift or altering
their official QC records. This is not a corrected GigaTIME inference run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import shift as nd_shift
import tifffile
import zarr


def score(values: np.ndarray, labels: np.ndarray, scorer_type: type) -> dict:
    scorer = scorer_type(values.shape)
    scorer.add(values, labels, 0, 0)
    return scorer.report()


def run(prepared: Path, inference: Path, qc_module_dir: Path,
        output: Path, dy: int, dx: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if (dy, dx) != (-3, 3):
        raise ValueError("Frozen train-derived diagnostic shift is (-3,+3) model pixels")
    sys.path.insert(0, str(qc_module_dir))
    from gigatime_nuclear_qc import NuclearAlignmentScore

    rows = {r["sample_id"]: r for r in
            json.loads((prepared / "PREP_COMPLETE.json").read_text())["records"]}
    records = []
    for marker in sorted(inference.glob("*/CASE_COMPLETE.json")):
        sample = marker.parent.name
        saved = json.loads(marker.read_text())
        frozen = json.loads((marker.parent / "stardist_qc.json").read_text())
        if saved["sample_id"] != sample or saved["model_mpp"] < .249999 or \
                saved["model_mpp"] > .250001 or \
                frozen["status"] != saved["nuclear_qc"]["status"]:
            raise ValueError(f"Saved output/QC identity changed: {sample}")
        store = zarr.open(str(marker.parent / "gigatime_output" /
                              "gigatime_probs.zarr"), mode="r")
        if store.attrs["gigatime"]["store_channels"][0] != "DAPI":
            raise ValueError(f"First saved channel is not DAPI: {sample}")
        dapi = np.asarray(store["0"][0], dtype=np.float32) / 65535.0
        original_labels = tifffile.imread(marker.parent / "stardist_labels.tif")
        yy = np.minimum(original_labels.shape[0]-1,
                        np.floor((np.arange(dapi.shape[0])+.5) *
                                 original_labels.shape[0] / dapi.shape[0]).astype(int))
        xx = np.minimum(original_labels.shape[1]-1,
                        np.floor((np.arange(dapi.shape[1])+.5) *
                                 original_labels.shape[1] / dapi.shape[1]).astype(int))
        labels = original_labels[yy[:, None], xx[None, :]]
        unchanged = score(dapi, labels, NuclearAlignmentScore)
        shifted = nd_shift(dapi, shift=(dy, dx), order=1, mode="constant",
                           cval=np.nan, prefilter=False)
        corrected = score(shifted, labels, NuclearAlignmentScore)
        records.append({"sample_id": sample, "organ": rows[sample]["organ"],
                        "split": rows[sample]["split"],
                        "frozen_status": frozen["status"],
                        "frozen_auc": frozen["auc"],
                        "unshifted_recomputed_status": unchanged["status"],
                        "unshifted_recomputed_auc": unchanged["auc"],
                        "fixed_shift_status": corrected["status"],
                        "fixed_shift_auc": corrected["auc"],
                        "fixed_shift_boundary_contrast": corrected["nuclear_boundary_contrast"],
                        "fixed_shift_spatial_shadow_status": corrected["spatial_shadow_status"],
                        "fixed_shift_border_shadow_status": corrected["border_shadow_status"]})
    report = {"status": "diagnostic_only", "cases": len(records),
              "train_selected_shift_model_px": [dy, dx], "model_mpp_um": .25,
              "unshifted_recomputed_vs_frozen_auc_max_abs_diff":
              max(abs(r["unshifted_recomputed_auc"]-r["frozen_auc"])
                  for r in records),
              "counts_by_split_and_status": dict(Counter(
                  f"{r['split']}:{r['fixed_shift_status']}" for r in records)),
              "records": records,
              "limit": "One translation selected from training images only, applied diagnostically to all completed fields. The scorer reads a whole field rather than tiled blocks; recomputation agreement must be checked. No QC record, marker store or classifier input was modified. A shift cannot rescue weak or biologically invalid GigaTIME markers."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--qc-module-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dy", type=int, default=-3)
    p.add_argument("--dx", type=int, default=3)
    a = p.parse_args()
    x = run(a.prepared, a.inference, a.qc_module_dir, a.output, a.dy, a.dx)
    print(json.dumps({k: x[k] for k in
                      ("cases", "counts_by_split_and_status",
                       "unshifted_recomputed_vs_frozen_auc_max_abs_diff")}))
