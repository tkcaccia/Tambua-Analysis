"""CPU-only gate between consecutive approved GigaTIME GPU images."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def read_table(path: Path, mask_name: str) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Empty quantification table: {path}")
    ids = set()
    for row in rows:
        cell_id = int(row["label_id"])
        if cell_id <= 0 or cell_id in ids:
            raise ValueError(f"Invalid or duplicate cell ID in {path}")
        if row["mask_name"] != mask_name or row["value_semantics"] != "uncalibrated_virtual_marker_score":
            raise ValueError(f"Incorrect marker semantics in {path}")
        if row["measured_protein_abundance"].lower() not in {"false", "0"}:
            raise ValueError(f"Virtual marker falsely declared measured: {path}")
        for marker in ("DAPI", "CK", "CD3", "CD68", "CD14", "CD16"):
            value = float(row[marker])
            if not 0 <= value <= 1:
                raise ValueError(f"Invalid {marker} score in {path}")
        ids.add(cell_id)
    return ids


def validate(input_dir: Path, output_dir: Path) -> dict:
    source = json.loads((input_dir / "provenance.json").read_text())
    inference = json.loads((output_dir / "gigatime_output/gigatime_metadata.json").read_text())
    if abs(float(inference["source_mpp"]) - float(source["mpp"])) > 1e-4:
        raise ValueError("GigaTIME source MPP differs from organizer metadata")
    if abs(float(inference["effective_mpp"]) - 0.25) > 1e-4:
        raise ValueError("GigaTIME did not use the required 0.25 µm/pixel grid")
    if inference["original_shape_yx"] != [source["height"], source["width"]]:
        raise ValueError("GigaTIME processed a different field of view")
    nuclei = read_table(output_dir / "quantification/output_nuclei_gigatime_mean_intensity.csv", "nuclei")
    cell = read_table(output_dir / "quantification/output_cell_gigatime_mean_intensity.csv", "cell")
    with (input_dir / "expert_types.csv").open(newline="", encoding="utf-8") as source_file:
        annotated = {int(row["label_id"]) for row in csv.DictReader(source_file)}
    if not nuclei <= annotated or not cell <= annotated:
        raise ValueError("Quantification includes an unannotated instance ID")
    if len(nuclei & cell) < 2:
        raise ValueError("Too few cells share nuclear and expanded-mask quantification")
    dapi = output_dir / "gigatime_output/jpg_image/DAPI.jpg"
    if not dapi.is_file() or not dapi.stat().st_size:
        raise ValueError("GigaTIME DAPI preview missing")
    report = {"sample_id": input_dir.name, "source_mpp": source["mpp"],
              "effective_mpp": inference["effective_mpp"],
              "expert_cells": len(annotated), "nuclei_quantified": len(nuclei),
              "cell_proxy_quantified": len(cell), "shared_quantified": len(nuclei & cell),
              "expert_cells_missing_any_quantification": len(annotated - (nuclei & cell))}
    target = output_dir / "VALIDATION.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(Path(sys.argv[1]), Path(sys.argv[2]))))
