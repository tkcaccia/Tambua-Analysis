#!/usr/bin/env python3
"""CPU-only integrity and descriptive summary of a completed HEMIT GPU bundle."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from audit_inputs import sha256


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "median": None, "q1": None, "q3": None}
    return {"n": len(array), "median": float(np.median(array)),
            "q1": float(np.quantile(array, .25)),
            "q3": float(np.quantile(array, .75))}


def summarize(bundle: Path) -> tuple[dict, list[dict]]:
    complete = json.loads((bundle / "BUNDLE_COMPLETE.json").read_text())
    if complete["status"] != "all_292_gpu_cases_complete_pending_visual_review" or \
       len(complete["cases"]) != 292:
        raise ValueError("HEMIT GPU bundle is incomplete; do not summarize partial data")
    all_rows = []
    elapsed = []
    for index, expected in enumerate(complete["cases"], start=1):
        case = bundle / f"case-{index:03d}"
        record = json.loads((case / "CASE_COMPLETE.json").read_text())
        seam = json.loads((case / "SEAM_QC.json").read_text())
        if record["index"] != index or record["id"] != expected["id"] or \
           record["status"] != "case_complete_pending_visual_review" or \
           record["rows"] != expected["rows"] or record["seconds"] != expected["seconds"] or \
           seam["status"] != "pass" or \
           sha256(case / "three_marker_probabilities.npz") != record["prediction_sha256"] or \
           not (case / "paired-contact.webp").exists():
            raise ValueError(f"Invalid or changed HEMIT case checkpoint {index}")
        elapsed.append(float(record["seconds"]))
        roi = record["id"].split("]_patch_")[0] + "]"
        for marker in record["rows"]:
            all_rows.append({"index": index, "image": record["id"], "roi_prefix": roi,
                             "measured": marker["measured"],
                             "virtual": marker["virtual"],
                             "spearman_rho": marker["spearman_rho"],
                             "auroc": marker["auroc"],
                             "average_precision": marker["average_precision"],
                             "positive_prevalence_gt_zero": marker["positive_prevalence_gt_zero"],
                             "seconds": record["seconds"]})
    marker_summary = []
    for name in ("Pan-CK", "CD3", "DAPI"):
        rows = [r for r in all_rows if r["measured"] == name]
        roi_groups = defaultdict(list)
        for row in rows:
            roi_groups[row["roi_prefix"]].append(float(row["spearman_rho"]))
        marker_summary.append({
            "measured": name,
            "images": len(rows),
            "roi_coordinate_prefixes_not_patient_ids": len(roi_groups),
            "patch_spearman": quantiles([float(r["spearman_rho"]) for r in rows]),
            "roi_median_spearman": quantiles([float(np.median(v)) for v in roi_groups.values()]),
            "patch_auroc": quantiles([float(r["auroc"]) for r in rows if r["auroc"] is not None]),
            "patch_average_precision": quantiles([float(r["average_precision"]) for r in rows if r["average_precision"] is not None]),
            "patch_positive_prevalence": quantiles([float(r["positive_prevalence_gt_zero"]) for r in rows]),
            "patches_auroc_above_half": sum(r["auroc"] is not None and r["auroc"] > .5 for r in rows),
        })
    report = {
        "status": "complete_technical_and_descriptive_summary_pending_all_image_visual_review",
        "dataset": "HEMIT official test / published 292 nonoverlapping, nonempty image pairs",
        "source_archive_sha256": "efe7dd5dc50ec0f789b0ce8a5258f5281a38001ac3dca81620f2a4be7041c943",
        "source_mpp_um": .25,
        "mpp_evidence": "MIPHEI-ViT study, because HEMIT TIFF pixel-size tags are absent",
        "images": 292, "technical_seam_qc_pass": 292,
        "distinct_roi_coordinate_prefixes_not_patient_ids": len({r["roi_prefix"] for r in all_rows}),
        "image_seconds": quantiles(elapsed), "total_inference_and_checkpoint_seconds": sum(elapsed),
        "markers": marker_summary,
        "interpretation_limits": [
            "Pan-CK/CD3 AUROC uses released mIHC pixels greater than zero as positive; the threshold is provisional.",
            "Patch scores are descriptive because patches within a source patient can be correlated.",
            "Coordinate prefixes are ROI locations, not verified independent patients; no patient-level confidence interval is claimed.",
            "One image was previously inspected as a CPU feasibility pilot; report a 291-image sensitivity if needed.",
            "GigaTIME probabilities and measured mIHC intensities have different semantics; rank correlation is a spatial diagnostic.",
            "DAPI nuclear localization does not establish CD3 or cytokeratin protein accuracy.",
        ],
    }
    return report, all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.csv.exists():
        raise FileExistsError("Summary output already exists")
    report, rows = summarize(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": report["status"], "images": report["images"],
                      "markers": report["markers"]}, indent=2))
