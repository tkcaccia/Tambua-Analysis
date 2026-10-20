#!/usr/bin/env python3
"""Quantify saved GigaTIME markers over StarDist cells, with independent labels.

This is CPU-only postprocessing. Expert MoNuSAC polygons provide cell types only
after a conservative, one-to-one match to detected nuclei; their heterogeneous
geometry is never used as the quantification compartment. All detected cells
are quantified so a separately fitted classifier can predict unlabelled cells.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile


GIGATIME_CHANNELS = (
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
    "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
    "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B",
    "Transgelin",
)
VALUE_SEMANTICS = "uncalibrated_virtual_marker_score"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_proxy(nuclei: np.ndarray, mpp: float, halo_um: float = 3.0) -> np.ndarray:
    """Assign a finite physical halo to its closest detected nucleus."""
    if nuclei.ndim != 2 or not np.issubdtype(nuclei.dtype, np.integer) or \
            np.any(nuclei < 0) or nuclei.max() == 0:
        raise ValueError("Expected a nonempty, nonnegative 2-D instance mask")
    if not np.isfinite(mpp) or mpp <= 0 or not np.isfinite(halo_um) or halo_um <= 0:
        raise ValueError("Pixel spacing and cell halo must be positive")
    distance, nearest = distance_transform_edt(
        nuclei == 0, sampling=(mpp, mpp), return_indices=True)
    owner = nuclei[nearest[0], nearest[1]]
    return np.where((nuclei > 0) | (distance <= halo_um), owner, 0).astype(np.uint32)


def validated_assignments(matched: dict, detected: np.ndarray) -> list[dict]:
    present = set(int(value) for value in np.unique(detected) if value > 0)
    if matched.get("assignments") is None:
        raise ValueError("Missing independent source-to-detection assignments")
    accepted = matched["assignments"]
    ids = [int(row["detected_id"]) for row in accepted]
    if len(ids) != len(set(ids)) or not set(ids) <= present:
        raise ValueError("Assignments are not unique detected cell IDs")
    if any(row["cell_type"] not in
           ("epithelial", "lymphocyte", "macrophage", "neutrophil")
           for row in accepted):
        raise ValueError("Unknown expert cell type")
    return accepted


def write_mask(path: Path, array: np.ndarray, mpp: float) -> None:
    tifffile.imwrite(path, array, ome=True, compression="zlib",
                     metadata={"axes": "YX", "PhysicalSizeX": mpp,
                               "PhysicalSizeY": mpp, "PhysicalSizeXUnit": "µm",
                               "PhysicalSizeYUnit": "µm"})


def load_inference_validation(case: Path, prepared: Path, sample: str) -> tuple[dict, Path]:
    """Accept either the original validated case or a checked CPU-only case."""
    original = case / "VALIDATION.json"
    if original.exists():
        return json.loads(original.read_text()), original
    checkpoint = case / "CASE_COMPLETE.json"
    row = json.loads(checkpoint.read_text())
    bundle = json.loads((case.parent / "BUNDLE_IDENTITY.json").read_text())
    if row.get("sample_id") != sample or row.get("source_rgb_sha256") != \
            sha256(prepared / "he.ome.tif") or row.get("channels") is None or \
            len(row["channels"]) != 23 or \
            bundle.get("model_revision") != "a6556030bdb8071857559e12b921620aa1bbac5b" or \
            bundle.get("device") != "CPU" or \
            abs(float(row.get("source_mpp", -1)) -
                float(json.loads((prepared / "provenance.json").read_text())["mpp"])) > 1e-6:
        raise ValueError("CPU GigaTIME case identity or 23-channel panel changed")
    return {"source_color_reference": "scanner_svs_rgb",
            "effective_mpp": row["model_mpp"]}, checkpoint


def quantify(quantifier: Path, image: Path, mask: Path, name: str,
             output: Path) -> None:
    prefix = output / f"detected_{name}"
    subprocess.run([sys.executable, str(quantifier), "--image", str(image),
                    "--mask", str(mask), "--mask-name", name,
                    "--out-quant-csv", str(prefix) + "_quantification.csv",
                    "--out-mean-csv", str(prefix) + "_mean_intensity.csv",
                    "--out-stats-csv", str(prefix) + "_intensity_stats.csv",
                    "--out-summary-json", str(prefix) + "_summary.json"], check=True)
    validate_quantified_table(prefix.with_name(prefix.name + "_mean_intensity.csv"),
                              prefix.with_name(prefix.name + "_summary.json"), name)


def validate_quantified_table(table_path: Path, summary_path: Path,
                              mask_name: str) -> int:
    """Reject old quantifiers that silently drop the virtual-score contract."""
    summary = json.loads(summary_path.read_text())
    semantics = summary.get("value_semantics", {})
    if summary.get("mask_name") != mask_name or \
            summary.get("channel_names") != list(GIGATIME_CHANNELS) or \
            semantics.get("value_type") != VALUE_SEMANTICS or \
            semantics.get("calibrated_probability") is not False or \
            semantics.get("measured_protein_abundance") is not False:
        raise ValueError(f"Quantifier omitted virtual-score provenance for {mask_name}")
    with table_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(("label_id", "mask_name",
                "value_semantics", "calibrated_probability",
                "measured_protein_abundance", *GIGATIME_CHANNELS)) <= \
                set(reader.fieldnames):
            raise ValueError(f"Quantifier omitted virtual-score columns for {mask_name}")
        rows = list(reader)
    ids = set()
    for row in rows:
        label_id = int(row["label_id"])
        if label_id <= 0 or label_id in ids or row["mask_name"] != mask_name or \
                row["value_semantics"] != VALUE_SEMANTICS or \
                row["calibrated_probability"].lower() not in ("false", "0") or \
                row["measured_protein_abundance"].lower() not in ("false", "0"):
            raise ValueError(f"Invalid quantified cell metadata for {mask_name}")
        ids.add(label_id)
        values = [float(row[channel]) for channel in GIGATIME_CHANNELS]
        if any(not math.isfinite(value) or value < 0 or value > 1
               for value in values):
            raise ValueError(f"Invalid virtual-marker score for {mask_name}")
    if len(rows) != summary.get("objects_quantified"):
        raise ValueError(f"Quantifier row count differs from summary for {mask_name}")
    return len(rows)


def run(sample: str, detected_path: Path, matched_path: Path,
        prepared: Path, gigatime_case: Path, quantifier: Path,
        output: Path, *, halo_um: float = 3.0,
        qc_report_path: Path | None = None) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This postprocessing must explicitly hide all GPUs")
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((prepared / "provenance.json").read_text())
    matched = json.loads(matched_path.read_text())
    validation, validation_path = load_inference_validation(gigatime_case, prepared, sample)
    qc_path = (qc_report_path if qc_report_path is not None else
               gigatime_case / "gigatime_output" /
               "output_gigatime_nuclear_qc.json")
    qc = json.loads(qc_path.read_text())
    fields = ("status", "alignment_status", "border_shadow_status",
              "spatial_shadow_status")
    if any(qc.get(name) != "pass" for name in fields):
        raise ValueError(f"GigaTIME DAPI/shadow QC failed for {sample}")
    if qc_report_path is not None and Path(qc.get("nuclei_mask", "")).resolve() != \
            detected_path.resolve():
        raise ValueError("Corrected DAPI QC did not use the supplied StarDist nuclei")
    if provenance.get("sample_id") != sample or matched.get("sample_id") != sample:
        raise ValueError("Sample identity differs across inputs")
    if provenance.get("source_color_reference") != "scanner_svs_rgb" or \
            validation.get("source_color_reference") != "scanner_svs_rgb" or \
            abs(float(validation.get("effective_mpp", -1)) - .25) > 1e-4:
        raise ValueError("Scanner RGB or calibrated GigaTIME scale unverified")
    nuclei = tifffile.imread(detected_path)
    with tifffile.TiffFile(prepared / "he.ome.tif") as source_tiff:
        source_shape = tuple(source_tiff.series[0].shape[:2])
    if nuclei.shape != source_shape:
        raise ValueError("StarDist labels and scanner image geometry differ")
    assignments = validated_assignments(matched, nuclei)
    expert = tifffile.imread(prepared / "expert_nuclei.ome.tif")
    if expert.shape != nuclei.shape:
        raise ValueError("Expert geometry and detected-cell grid differ")
    expert_areas = np.bincount(expert.ravel())
    if any(int(row["expert_id"]) >= len(expert_areas) or
           expert_areas[int(row["expert_id"])] == 0 for row in assignments):
        raise ValueError("Matched expert region is absent from source mask")
    mpp = float(provenance["mpp"])
    cells = cell_proxy(nuclei, mpp, halo_um)
    if not np.array_equal(cells[nuclei > 0], nuclei[nuclei > 0]):
        raise ValueError("Cell proxy displaced a detected nucleus")
    output.mkdir(parents=True)
    nuclear_mask = output / "stardist_nuclei.ome.tif"
    cell_mask = output / "stardist_cell_proxy.ome.tif"
    write_mask(nuclear_mask, nuclei.astype(np.uint32), mpp)
    write_mask(cell_mask, cells, mpp)
    with (output / "expert_types_on_detected.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label_id", "cell_type",
                                                     "expert_id", "matching_rule",
                                                     "expert_area_native_px"))
        writer.writeheader()
        writer.writerows({"label_id": row["detected_id"],
                          "cell_type": row["cell_type"],
                          "expert_id": row["expert_id"],
                          "matching_rule": row["rule"],
                          "expert_area_native_px":
                              int(expert_areas[int(row["expert_id"])])}
                         for row in assignments)
    image = gigatime_case / "gigatime_output"
    for mask, name in ((nuclear_mask, "nuclei"), (cell_mask, "cell")):
        quantify(quantifier, image, mask, name, output)
    summaries = [json.loads((output / f"detected_{name}_summary.json").read_text())
                 for name in ("nuclei", "cell")]
    detected_count = int(np.count_nonzero(np.bincount(nuclei.ravel())[1:]))
    if any(item["objects_quantified"] != detected_count or
           len(item["channel_names"]) != 23 for item in summaries):
        raise ValueError("Detected cells or the full 23-marker panel were lost")
    result = {"sample_id": sample, "status": "complete",
              "device": "CPU", "source_color_reference": "scanner_svs_rgb",
              "gigatime_effective_mpp": validation["effective_mpp"],
              "source_mpp": mpp, "halo_um": halo_um,
              "detected_cells_quantified": detected_count,
              "expert_types_conservatively_matched": len(assignments),
              "unlabelled_detected_cells": detected_count - len(assignments),
              "source_polygon_roles": "cell type only; never a quantification mask",
              "detected_mask_sha256": sha256(detected_path),
              "source_provenance_sha256": sha256(prepared / "provenance.json"),
              "matching_sha256": sha256(matched_path),
              "nuclear_qc_reference": ("stardist_detected_nuclei" if qc_report_path is not None
                                        else "original_case_reference"),
              "nuclear_qc_sha256": sha256(qc_path),
              "gigatime_validation_sha256": sha256(validation_path)}
    result["quantifier_sha256"] = sha256(quantifier)
    (output / "CASE_COMPLETE.json.tmp").write_text(json.dumps(result, indent=2) + "\n")
    (output / "CASE_COMPLETE.json.tmp").replace(output / "CASE_COMPLETE.json")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--detected", type=Path, required=True)
    ap.add_argument("--matched", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--gigatime-case", type=Path, required=True)
    ap.add_argument("--quantifier", type=Path, required=True)
    ap.add_argument("--qc-report", type=Path,
                    help="Corrected DAPI QC report using the supplied StarDist mask")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.sample, args.detected, args.matched,
                         args.prepared, args.gigatime_case,
                         args.quantifier, args.output,
                         qc_report_path=args.qc_report), indent=2))
