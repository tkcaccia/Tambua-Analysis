#!/usr/bin/env python3
"""Fail-closed technical validation of a saved ORION full-panel rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr

CHANNELS = ("DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
            "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
            "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(scope: dict, patient: str, input_case: Path, previous_case: Path,
             current_case: Path) -> dict:
    entry = next((x for x in scope["cases"] if x["patient"] == patient), None)
    if entry is None or any(p.name != patient for p in (input_case, previous_case, current_case)):
        raise ValueError("Patient identity does not match frozen scope")
    for name, expected in entry["input_sha256"].items():
        if digest(input_case / name) != expected:
            raise ValueError(f"Changed source: {patient}/{name}")
    if digest(previous_case / "CASE_COMPLETE.json") != entry["prior_complete_sha256"]:
        raise ValueError("Changed prior technical QC record")
    provenance = json.loads((input_case / "PROVENANCE.json").read_text())
    if any(abs(float(provenance[k]) - .325) > 1e-6 for k in ("mpp_x", "mpp_y")):
        raise ValueError("Physical source resolution changed")
    with tifffile.TiffFile(input_case / "he.ome.tif") as tif:
        page = tif.pages[0]
        if page.photometric != tifffile.PHOTOMETRIC.RGB or page.shape != (2048, 2048, 3):
            raise ValueError("Unexpected scanner RGB source")
    prior = json.loads((previous_case / "CASE_COMPLETE.json").read_text())
    if prior.get("patient") != patient or prior.get("technical_qc_passed") is not True:
        raise ValueError("Prior one-channel technical QC did not pass")
    before = zarr.open(str(previous_case / "gigatime" / "gigatime_probs.zarr"), mode="r")["0"]
    store = zarr.open(str(current_case / "gigatime" / "gigatime_probs.zarr"), mode="r")
    after = store["0"]
    meta = store.attrs["gigatime"]
    if before.shape[0] != 1 or after.shape[0] != 23 or before.shape[1:] != after.shape[1:] or \
       before.dtype != np.uint16 or after.dtype != np.uint16 or \
       tuple(meta["store_channels"]) != CHANNELS or \
       abs(float(meta["source_mpp"]) - .325) > 1e-6 or \
       abs(float(meta["effective_mpp"]) - .25) > 1e-6:
        raise ValueError("Saved full-channel geometry or channel order invalid")
    seam = json.loads((current_case / "gigatime" / "gigatime_seam_qc.json").read_text())
    marker = json.loads((current_case / "gigatime" / "gigatime_marker_score_qc.json").read_text())
    if seam.get("status") != "pass" or marker.get("status") not in ("pass", "review"):
        raise ValueError("Seam or marker-distribution technical QC failed")
    summaries = list((current_case / "quantification").glob(
        "*_gigatime_integrated_quantification_summary.json"))
    if len(summaries) != 1:
        raise ValueError("Expected one integrated quantification summary")
    quant = json.loads(summaries[0].read_text())
    if quant.get("technical_qc_passed") is not True:
        raise ValueError("Virtual-DAPI technical QC did not pass")
    first, second = np.asarray(before[0], dtype=np.int32), np.asarray(after[0], dtype=np.int32)
    delta = np.abs(first - second)
    corr = float(np.corrcoef(first.ravel(), second.ravel())[0, 1])
    p99 = float(np.quantile(delta, .99))
    if not np.isfinite(corr) or corr < .99 or p99 > 3000:
        raise ValueError(f"Rerun DAPI changed materially: r={corr:.6f}, p99={p99:.1f}")
    return {"status": "technical_complete_pending_visual_and_protein_validation",
            "patient": patient, "source_rgb_sha256": entry["input_sha256"]["he.ome.tif"],
            "source_mpp": .325, "model_mpp": .25, "model_shape_yx": list(after.shape[1:]),
            "channels": list(CHANNELS), "seam_qc": seam["status"],
            "marker_distribution_qc": marker["status"],
            "nuclear_qc_passed": True, "old_vs_new_dapi_pearson": corr,
            "old_vs_new_dapi_p99_absolute_uint16_difference": p99,
            "limit": "Technical completion does not validate protein expression or cell-type classification."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scope", type=Path, required=True)
    p.add_argument("--patient", required=True)
    p.add_argument("--input-case", type=Path, required=True)
    p.add_argument("--previous-case", type=Path, required=True)
    p.add_argument("--current-case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = validate(json.loads(a.scope.read_text()), a.patient, a.input_case,
                      a.previous_case, a.current_case)
    if a.output.exists():
        raise FileExistsError(a.output)
    temporary = a.output.with_suffix(a.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(a.output)
    print(json.dumps({"patient": a.patient, "status": result["status"],
                      "dapi_pearson": result["old_vs_new_dapi_pearson"]}))
