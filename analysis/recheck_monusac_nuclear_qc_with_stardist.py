#!/usr/bin/env python3
"""Reassess saved virtual DAPI against actual H&E-derived StarDist nuclei.

Uses the original CellPhenotyper QC implementation and unchanged thresholds.
This is CPU-only; the source MoNuSAC polygons are not nuclear references.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This QC reassessment must explicitly hide all GPUs")


FIELDS = ("status", "alignment_status", "border_shadow_status",
          "spatial_shadow_status")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--detected-fit", type=Path, required=True)
    ap.add_argument("--detected-eval", type=Path, required=True)
    ap.add_argument("--qc-script", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cohort = json.loads((args.source / "input" / "COHORT.json").read_text())
    fold = json.loads(args.fold.read_text())
    if len(cohort["records"]) != 120 or not \
            (args.source / "evidence" / "BUNDLE_COMPLETE.json").exists():
        raise ValueError("Corrected-RGB source bundle is incomplete")
    dev = set(fold["development_patients"])
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for row in cohort["records"]:
        sample = row["sample_id"]
        split = ("test" if row["split"] == "test" else
                 "development" if row["patient_id"] in dev else "fit")
        root = args.detected_fit if split == "fit" else args.detected_eval
        mask = root / sample / "stardist_labels.tif"
        old = args.source / "output" / sample / "gigatime_output" / "output_gigatime_nuclear_qc.json"
        image = args.source / "output" / sample / "gigatime_output"
        report = args.output / f"{sample}_nuclear_qc.json"
        subprocess.run([sys.executable, str(args.qc_script),
                        "--image", str(image), "--nuclei", str(mask),
                        "--output", str(report), "--mode", "warn"], check=True,
                       stdout=subprocess.DEVNULL)
        original = json.loads(old.read_text())
        corrected = json.loads(report.read_text())
        if corrected["nuclei_mask"] != str(mask.resolve()) or \
                corrected["image_shape_yx"] != original["image_shape_yx"]:
            raise ValueError(f"Reference or inference-grid mismatch: {sample}")
        summary.append({"sample_id": sample, "patient_id": row["patient_id"],
                        "organ": row["organ"], "split": split,
                        "source_polygon_qc": {name: original.get(name) for name in FIELDS},
                        "stardist_nuclear_qc": {name: corrected.get(name) for name in FIELDS},
                        "source_polygon_auc": original.get("auc"),
                        "stardist_nuclear_auc": corrected.get("auc"),
                        "stardist_nuclear_boundary_contrast":
                            corrected.get("nuclear_boundary_contrast")})
    count = Counter((row["split"], row["stardist_nuclear_qc"]["status"])
                    for row in summary)
    result = {"status": "complete", "device": "CPU",
              "reference": "StarDist2D 2D_versatile_he nuclei on scanner RGB",
              "qc_implementation": str(args.qc_script),
              "thresholds": "unchanged from the original corrected-RGB bundle",
              "cases": len(summary),
              "by_split_status": {f"{split}:{status}": n for (split, status), n in sorted(count.items())},
              "records": summary}
    staged = args.output / "COHORT_COMPLETE.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(args.output / "COHORT_COMPLETE.json")
    print(json.dumps({"cases": len(summary), "by_split_status": result["by_split_status"]}, indent=2))


if __name__ == "__main__":
    main()
