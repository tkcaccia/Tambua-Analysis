#!/usr/bin/env python3
"""CPU-only fixed compartment correlations on ten ORION full-panel reruns."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from score_orion_raw_nine import PAIRS, PATIENTS


def load_case(case: Path, compartment: str) -> pd.DataFrame:
    paths = list((case / "quantification").glob(
        f"*_{compartment}_gigatime_intensity_stats.csv"))
    if len(paths) != 1:
        raise ValueError(f"Missing unique {compartment} quantification: {case}")
    frame = pd.read_csv(paths[0]).set_index("label_id")
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError("Invalid quantified cell IDs")
    return frame


def rho(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 30 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def run(study: Path, pilot: Path, nine: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Cell quantification analysis must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    counts = []
    for patient in PATIENTS:
        case = (pilot if patient == "CRC04" else nine) / patient
        technical = json.loads((case / "TECHNICAL_COMPLETE.json").read_text())
        if technical.get("patient") != patient or \
           technical.get("status") != "technical_complete_pending_visual_and_protein_validation":
            raise ValueError(f"Unvalidated full-panel case: {patient}")
        measured = pd.read_csv(study / "cpu-prep" / patient / "measured_cells.csv").set_index("CellID")
        tables = {comp: load_case(case, comp) for comp in ("cell", "cyto", "nuclei")}
        identities = set(measured.index)
        for table in tables.values():
            identities &= set(table.index)
        ids = sorted(identities)
        if len(ids) < 30:
            raise ValueError(f"Insufficient common published cells: {patient}")
        counts.append({"patient": patient, "common_cells": len(ids)})
        for marker, (measured_name, mapping) in PAIRS.items():
            observed = pd.to_numeric(measured.loc[ids, measured_name], errors="coerce").to_numpy(float)
            for compartment, frame in tables.items():
                virtual = pd.to_numeric(frame.loc[ids, f"{marker}__mean"], errors="coerce").to_numpy(float)
                finite = np.isfinite(observed) & np.isfinite(virtual)
                rows.append({"patient": patient, "virtual": marker, "measured": measured_name,
                             "mapping": mapping, "compartment": compartment,
                             "cells": int(finite.sum()),
                             "cell_spearman_rho": rho(observed[finite], virtual[finite])})
    summary = []
    for marker, (measured_name, mapping) in PAIRS.items():
        for compartment in ("cell", "cyto", "nuclei"):
            values = [r["cell_spearman_rho"] for r in rows if r["virtual"] == marker
                      and r["compartment"] == compartment and r["cell_spearman_rho"] is not None]
            summary.append({"virtual": marker, "measured": measured_name, "mapping": mapping,
                            "compartment": compartment, "patients": len(values),
                            "patient_median_rho": float(np.median(values)) if values else None,
                            "positive_patients": int(sum(x > 0 for x in values))})
    report = {"status": "fixed_compartment_cell_association_pending_biological_review",
              "patients": list(PATIENTS), "counts": counts, "per_patient": rows,
              "equal_patient_summary": summary,
              "method": "Published ORION cell IDs matched to CellPhenotyper quantified full-panel GigaTIME means; no compartment optimized on these evaluation patients.",
              "limitation": "Cell Spearman is not a marker-positive classification score and does not validate single-cell protein abundance. One field per patient and no independent tissue-domain calibration."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--nine", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.study, a.pilot, a.nine, a.output)
    print(json.dumps({"patients": len(result["patients"]),
                      "summary": result["equal_patient_summary"]}, indent=2))
