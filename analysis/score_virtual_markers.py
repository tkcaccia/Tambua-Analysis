"""Compare same-cell virtual GigaTIME scores with ORION measured proteins.

This is a single-patient descriptive check, not a cell-type classification or
held-out biological validation result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAIRS = {"DAPI": "Hoechst", "CK": "Pan-CK", "CD3": "CD3e", "CD68": "CD68",
         "CD4": "CD4", "CD20": "CD20", "CD8": "CD8a", "PD-L1": "PD-L1",
         "PD-1": "PD-1", "Ki67": "Ki67"}


def score(virtual_csv: Path, measured_csv: Path) -> dict:
    virtual = pd.read_csv(virtual_csv).set_index("label_id")
    measured = pd.read_csv(measured_csv).set_index("CellID")
    if not virtual.index.is_unique or not measured.index.is_unique:
        raise ValueError("Cell identities must be unique in both tables")
    if set(virtual.index) != set(measured.index) or len(virtual) != 4104:
        raise ValueError("Expected exactly 4,104 matched ORION cell identities")
    measured = measured.loc[virtual.index]
    correlations = []
    for virtual_name, measured_name in PAIRS.items():
        rho = virtual[virtual_name].corr(measured[measured_name], method="spearman")
        correlations.append({"virtual": virtual_name, "measured": measured_name,
                             "spearman_rho": round(float(rho), 4)})
    return {"sample_id": "ORION CRC33_01", "patient_count": 1,
            "matched_cells": len(virtual), "metric": "cell-level Spearman correlation",
            "interpretation": "Descriptive same-section marker association; not cell-type accuracy or an independent-patient test.",
            "pairs": correlations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--virtual", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.virtual, args.measured)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
