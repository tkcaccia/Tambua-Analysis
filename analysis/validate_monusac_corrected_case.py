"""Validate one corrected-RGB MoNuSAC inference without accepting failed QC."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

CHANNELS = (
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
    "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
    "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B",
    "Transgelin",
)


def table_ids(path: Path, expected_mask: str) -> set[int]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows or not set(CHANNELS).issubset(rows[0]):
        raise ValueError(f"Missing cells or marker channels: {path}")
    ids: set[int] = set()
    for row in rows:
        label_id = int(row["label_id"])
        if label_id <= 0 or label_id in ids or row["mask_name"] != expected_mask:
            raise ValueError(f"Invalid {expected_mask} row: {path}")
        if row["value_semantics"] != "uncalibrated_virtual_marker_score":
            raise ValueError(f"Unexpected marker units: {path}")
        for name in CHANNELS:
            value = float(row[name])
            if not 0 <= value <= 1:
                raise ValueError(f"Invalid {name} score: {path}")
        ids.add(label_id)
    return ids


def validate(prepared: Path, output: Path) -> dict:
    provenance = json.loads((prepared / "provenance.json").read_text())
    if provenance.get("source_color_reference") != "scanner_svs_rgb":
        raise ValueError("Input is not the corrected scanner RGB image")
    metadata = json.loads((output / "gigatime_output/gigatime_metadata.json").read_text())
    if abs(float(metadata["source_mpp"]) - float(provenance["mpp"])) > 1e-4:
        raise ValueError("Source pixel spacing differs from annotation")
    if abs(float(metadata["effective_mpp"]) - 0.25) > 1e-4:
        raise ValueError("GigaTIME did not use the 0.25 µm/pixel model grid")
    if metadata["original_shape_yx"] != [provenance["height"], provenance["width"]]:
        raise ValueError("Inference field differs from the annotated image")
    qc = json.loads((output / "gigatime_output/output_gigatime_nuclear_qc.json").read_text())
    if qc.get("status") not in {"pass", "fail"}:
        raise ValueError("Missing or malformed nuclear QC status")
    summary = json.loads((output / "quantification/output_gigatime_integrated_quantification_summary.json").read_text())
    if summary.get("technical_qc_passed") is not (qc["status"] == "pass"):
        raise ValueError("Quantification and QC status disagree")
    nuclei = table_ids(output / "quantification/output_nuclei_gigatime_mean_intensity.csv", "nuclei")
    cells = table_ids(output / "quantification/output_cell_gigatime_mean_intensity.csv", "cell")
    with (prepared / "expert_types.csv").open(newline="", encoding="utf-8") as source:
        expert = {int(row["label_id"]) for row in csv.DictReader(source)}
    if not nuclei <= expert or not cells <= expert or len(nuclei & cells) < 2:
        raise ValueError("Quantified cells do not match expert IDs")
    if not (output / "gigatime_output/gigatime_probs.zarr").is_dir():
        raise ValueError("Native marker store missing")
    result = {
        "sample_id": prepared.name,
        "source_color_reference": provenance["source_color_reference"],
        "source_mpp": provenance["mpp"],
        "effective_mpp": metadata["effective_mpp"],
        "expert_regions": len(expert),
        "nuclei_quantified": len(nuclei),
        "cells_quantified": len(cells),
        "matched_cells": len(nuclei & cells),
        "nuclear_qc": qc["status"],
        "classifiable": qc["status"] == "pass",
    }
    (output / "VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(validate(Path(sys.argv[1]), Path(sys.argv[2]))))
