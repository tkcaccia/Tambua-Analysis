#!/usr/bin/env python3
"""CPU-only postprocessing of existing QC-passing MoNuSAC GigaTIME cases.

The saved GigaTIME predictions are read; no model inference or GPU launcher is
called. Existing, complete cases may be reused by identity, and any error stops
the batch without processing a later case.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU postprocessing must explicitly hide all GPUs")

from quantify_detected_monusac_cells import run


def quality_pass(case: Path) -> bool:
    qc = json.loads((case / "gigatime_output" /
                     "output_gigatime_nuclear_qc.json").read_text())
    return all(qc.get(name) == "pass" for name in
               ("status", "alignment_status", "border_shadow_status",
                "spatial_shadow_status"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--detected-fit", type=Path, required=True)
    ap.add_argument("--detected-eval", type=Path, required=True)
    ap.add_argument("--matched-fit", type=Path, required=True)
    ap.add_argument("--matched-eval", type=Path, required=True)
    ap.add_argument("--quantifier", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cohort = json.loads((args.source / "input" / "COHORT.json").read_text())
    fold = json.loads(args.fold.read_text())
    if len(cohort["records"]) != 120 or not \
            (args.source / "evidence" / "BUNDLE_COMPLETE.json").is_file():
        raise ValueError("The corrected-RGB source bundle is incomplete")
    dev_patients = set(fold["development_patients"])
    selected = [row for row in cohort["records"]
                if quality_pass(args.source / "output" / row["sample_id"])]
    if len(selected) != 48:
        raise ValueError(f"Strict QC pass count changed: {len(selected)}")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "COHORT_COMPLETE.json").exists():
        raise FileExistsError("CPU postprocessing cohort already complete")
    results = []
    for row in selected:
        sample = row["sample_id"]
        fit = row["split"] == "train" and row["patient_id"] not in dev_patients
        detected_root = args.detected_fit if fit else args.detected_eval
        matched_root = args.matched_fit if fit else args.matched_eval
        output = args.output / sample
        if output.exists():
            checkpoint = output / "CASE_COMPLETE.json"
            if not checkpoint.exists():
                raise ValueError(f"Incomplete existing CPU case: {sample}")
            result = json.loads(checkpoint.read_text())
            if result.get("sample_id") != sample or result.get("status") != "complete":
                raise ValueError(f"Invalid existing CPU checkpoint: {sample}")
        else:
            result = run(sample, detected_root / sample / "stardist_labels.tif",
                         matched_root / f"{sample}.json", args.source / "prepared" / sample,
                         args.source / "output" / sample, args.quantifier, output)
        results.append(dict(sample_id=sample, patient_id=row["patient_id"],
                            split=("test" if row["split"] == "test" else
                                   "fit" if fit else "development"),
                            organ=row["organ"],
                            detected_cells=result["detected_cells_quantified"],
                            expert_matched_cells=result["expert_types_conservatively_matched"]))
    if len(results) != 48:
        raise ValueError("Not all QC-passing cases completed")
    report = {"status": "complete", "device": "CPU",
              "scope": "existing corrected-RGB and strict-QC-passing MoNuSAC cases only",
              "cases": len(results),
              "by_split": dict(Counter(row["split"] for row in results)),
              "detected_cells": sum(row["detected_cells"] for row in results),
              "expert_matched_cells": sum(row["expert_matched_cells"] for row in results),
              "records": results}
    temp = args.output / "COHORT_COMPLETE.json.tmp"
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(args.output / "COHORT_COMPLETE.json")
    print(json.dumps({key: report[key] for key in
                      ("cases", "by_split", "detected_cells", "expert_matched_cells")}, indent=2))


if __name__ == "__main__":
    main()
