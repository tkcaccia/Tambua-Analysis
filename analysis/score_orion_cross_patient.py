#!/usr/bin/env python3
"""Equal-patient pooled, within-patient rank association on reserved ORION cells.

This CPU-only analysis uses fixed whole-cell virtual means and the published
same-cell ORION whole-cell measurements. Within-patient percentile ranks remove
between-slide offsets before pooling; each patient contributes equal weight.
Patients, not cells, are resampled for uncertainty. This does not score raw
spatial fluorescence images or establish virtual-protein calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


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


def moments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) != len(y) or len(x) < 30 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("At least 30 finite same-cell pairs are required")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Constant marker values cannot be ranked")
    a = rankdata(x, method="average") / (len(x) + 1)
    b = rankdata(y, method="average") / (len(y) + 1)
    return np.array([a.mean(), b.mean(), np.mean(a * a),
                     np.mean(b * b), np.mean(a * b)], dtype=np.float64)


def correlation_from_moments(values: np.ndarray) -> float:
    avg = np.asarray(values, dtype=np.float64).mean(axis=0)
    covariance = avg[4] - avg[0] * avg[1]
    vx = avg[2] - avg[0] ** 2
    vy = avg[3] - avg[1] ** 2
    if vx <= 0 or vy <= 0:
        raise ValueError("Pooled within-patient rank variance is zero")
    return float(covariance / np.sqrt(vx * vy))


def patient_ids(summary: dict) -> list[str]:
    rows = [row for row in summary["patient_records"]
            if row["split"] == "evaluation" and row["technical_qc_passed"]]
    ids = [row["patient"] for row in rows]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("Expected ten distinct reserved, technical-QC-passing patients")
    return sorted(ids)


def patient_vectors(root: Path, patient: str, summary_row: dict) -> dict:
    case = root / "gpu-output" / patient
    saved = json.loads((case / "CASE_COMPLETE.json").read_text())
    seam = json.loads((case / "gigatime" / "gigatime_seam_qc.json").read_text())
    if not saved["technical_qc_passed"] or seam["status"] != "pass":
        raise ValueError(f"Recorded virtual DAPI/seam QC fails for {patient}")
    tables = {}
    for compartment in ("cell", "cyto", "nuclei"):
        paths = list((case / "quantification").glob(
            f"*_{compartment}_gigatime_intensity_stats.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one {compartment} quantification table: {patient}")
        tables[compartment] = pd.read_csv(paths[0]).set_index("label_id")
    predicted = tables["cell"]
    measured = pd.read_csv(root / "cpu-prep" / patient / "measured_cells.csv").set_index("CellID")
    if not measured.index.is_unique or measured.index.hasnans or any(
            not frame.index.is_unique or frame.index.hasnans for frame in tables.values()):
        raise ValueError(f"Duplicate or missing cell identities: {patient}")
    if any(not set(frame.index) <= set(measured.index) for frame in tables.values()):
        raise ValueError(f"Predicted cell IDs absent from measured data: {patient}")
    common = set(measured.index)
    for frame in tables.values():
        common &= set(frame.index)
    ids = sorted(common)
    if len(ids) != summary_row["common_cells"]:
        raise ValueError(f"Common-cell count changed for {patient}")
    predicted = predicted.loc[ids]
    measured = measured.loc[ids]
    result = {}
    for virtual, (target, mapping) in PAIRS.items():
        x = pd.to_numeric(predicted[f"{virtual}__mean"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(measured[target], errors="coerce").to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 30:
            raise ValueError(f"Too few finite same-cell values: {patient}/{virtual}")
        x, y = x[finite], y[finite]
        result[virtual] = {"patient": patient, "cells": len(x), "mapping": mapping,
                           "measured": target, "moments": moments(x, y),
                           "within_patient_spearman": float(spearmanr(x, y).statistic)}
    return result


def run(root: Path, published_summary: Path, output: Path,
        bootstrap_replicates: int = 10000, seed: int = 20261020) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if bootstrap_replicates < 1000:
        raise ValueError("At least 1000 patient bootstrap replicates are required")
    summary = json.loads(published_summary.read_text())
    ids = patient_ids(summary)
    by_id = {row["patient"]: row for row in summary["patient_records"]}
    data = {patient: patient_vectors(root, patient, by_id[patient]) for patient in ids}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), size=(bootstrap_replicates, len(ids)))
    rows = []
    for virtual, (target, mapping) in PAIRS.items():
        matrix = np.stack([data[patient][virtual]["moments"] for patient in ids])
        observed = correlation_from_moments(matrix)
        sampled = matrix[draw].mean(axis=1)
        covariance = sampled[:, 4] - sampled[:, 0] * sampled[:, 1]
        vx = sampled[:, 2] - sampled[:, 0] ** 2
        vy = sampled[:, 3] - sampled[:, 1] ** 2
        boot = covariance / np.sqrt(vx * vy)
        if not np.isfinite(boot).all():
            raise ValueError(f"Nonfinite patient bootstrap statistic: {virtual}")
        rows.append({"virtual": virtual, "measured": target, "mapping": mapping,
                     "patients": len(ids),
                     "same_cells_total": sum(data[p][virtual]["cells"] for p in ids),
                     "equal_patient_pooled_within_patient_rank_correlation": observed,
                     "patient_bootstrap_95pct_ci": [float(v) for v in np.quantile(boot, [.025, .975])],
                     "per_patient": [{"patient": p, "cells": data[p][virtual]["cells"],
                                      "spearman_rho": data[p][virtual]["within_patient_spearman"]}
                                     for p in ids]})
    report = {"status": "complete", "device": "CPU", "patient_ids": ids,
              "patients": len(ids), "marker_pairs": len(rows),
              "estimator": "Within each patient, rank each marker across the same measured cells and scale ranks by n+1; pool patient-specific five moments with equal patient weight; compute Pearson correlation of the pooled within-patient ranks. Resample whole patients for the 95% percentile CI.",
              "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
              "source_summary": str(published_summary),
              "limit": "One QC-passing tissue window per reserved patient; whole-cell virtual means versus published whole-cell measurements. This is not raw spatial IF validation, intensity calibration, or whole-slide validation. Related target pairs are distinguished from exact pairs.",
              "results": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--published-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.published_summary, args.output)
    print(json.dumps({"patients": result["patients"],
                      "marker_pairs": result["marker_pairs"],
                      "pooled_rank_rho": {r["virtual"]: r["equal_patient_pooled_within_patient_rank_correlation"] for r in result["results"]}}))
