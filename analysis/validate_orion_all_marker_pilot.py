#!/usr/bin/env python3
"""Verify one saved 23-channel ORION GigaTIME pilot before any visual review.

This does not score proteins or mark them biologically valid. It checks source,
resolution, channel order, technical QC, and agreement of rerun DAPI with the
previously saved one-channel output on the exact same ORION field.
"""
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
            "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B",
            "Transgelin")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_case: Path, previous_case: Path, pilot_case: Path,
        output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((input_case / "PROVENANCE.json").read_text())
    if input_case.name != "CRC04" or previous_case.name != "CRC04" or \
            pilot_case.name != "CRC04" or \
            abs(float(provenance["mpp_x"]) - .325) > 1e-6 or \
            abs(float(provenance["mpp_y"]) - .325) > 1e-6:
        raise ValueError("Wrong ORION patient or physical resolution")
    source = input_case / "he.ome.tif"
    with tifffile.TiffFile(source) as tif:
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB or \
                tif.pages[0].shape != (2048, 2048, 3):
            raise ValueError("Source is not the expected registered RGB field")
    prior = json.loads((previous_case / "CASE_COMPLETE.json").read_text())
    if prior["patient"] != "CRC04" or not prior["technical_qc_passed"]:
        raise ValueError("Previous CRC04 technical QC not passed")
    old = zarr.open(str(previous_case / "gigatime" / "gigatime_probs.zarr"), mode="r")
    new = zarr.open(str(pilot_case / "gigatime" / "gigatime_probs.zarr"), mode="r")
    before, after = old["0"], new["0"]
    meta = new.attrs["gigatime"]
    if before.shape[0] != 1 or after.shape[0] != 23 or \
            tuple(after.shape[1:]) != tuple(before.shape[1:]) or \
            before.dtype != np.uint16 or after.dtype != np.uint16 or \
            tuple(meta["store_channels"]) != CHANNELS or \
            abs(float(meta["source_mpp"]) - .325) > 1e-6 or \
            abs(float(meta["effective_mpp"]) - .25) > 1e-6:
        raise ValueError("Saved full-channel marker grid does not match the source")
    seam = json.loads((pilot_case / "gigatime" / "gigatime_seam_qc.json").read_text())
    marker = json.loads((pilot_case / "gigatime" / "gigatime_marker_score_qc.json").read_text())
    if seam["status"] != "pass" or marker["status"] not in {"pass", "review"}:
        raise ValueError("Full-channel seam or distribution QC failed")
    summaries = list((pilot_case / "quantification").glob(
        "*_gigatime_integrated_quantification_summary.json"))
    if len(summaries) != 1:
        raise ValueError("Missing unique integrated quantification report")
    quant = json.loads(summaries[0].read_text())
    if not quant.get("technical_qc_passed"):
        raise ValueError("Rerun DAPI technical QC failed")
    prior_dapi = np.asarray(before[0], dtype=np.int32)
    rerun_dapi = np.asarray(after[0], dtype=np.int32)
    difference = np.abs(prior_dapi - rerun_dapi)
    corr = float(np.corrcoef(prior_dapi.ravel(), rerun_dapi.ravel())[0, 1])
    p99 = float(np.quantile(difference, .99))
    if not np.isfinite(corr) or corr < .99 or p99 > 3000:
        raise ValueError(f"DAPI rerun differs materially: r={corr:.6f}, p99={p99:.1f}")
    report = {"status": "technical_complete_pending_visual_and_protein_validation",
              "patient": "CRC04", "input_rgb_sha256": digest(source),
              "source_mpp": .325, "model_mpp": .25,
              "model_shape_yx": list(after.shape[1:]),
              "channels": list(CHANNELS),
              "full_channel_seam_qc": seam["status"],
              "marker_distribution_qc": marker["status"],
              "marker_distribution_warnings": marker.get("warnings", []),
              "rerun_nuclear_qc": quant["technical_qc_passed"],
              "old_vs_new_dapi_pearson": corr,
              "old_vs_new_dapi_p99_absolute_uint16_difference": p99,
              "old_vs_new_dapi_exact_pixel_fraction": float(np.mean(difference == 0)),
              "limit": "One previously QC-passing ORION window. Technical success does not validate any protein marker, cross-patient correlation, or cell-type classification."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-case", type=Path, required=True)
    parser.add_argument("--previous-case", type=Path, required=True)
    parser.add_argument("--pilot-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input_case, args.previous_case, args.pilot_case, args.output)
    print(json.dumps({"patient": report["patient"], "status": report["status"],
                      "old_vs_new_dapi_pearson": report["old_vs_new_dapi_pearson"]}))
