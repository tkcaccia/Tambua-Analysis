#!/usr/bin/env python3
"""Check the nuclear QC gate against the authors' eightfold-coarsened target.

This CPU-only self-mask control never runs GigaTIME. It tests whether a perfect
predictor of the paper's deliberately blurred target would pass the existing
QC thresholds; it cannot estimate real model accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
import tifffile
import torch
import torch.nn.functional as F


def run(inference: Path, qc_module_dir: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Coarse-target audit must hide all GPUs")
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(qc_module_dir))
    from gigatime_nuclear_qc import NuclearAlignmentScore

    torch.set_num_threads(2)
    cases = sorted(inference.glob("*/CASE_COMPLETE.json"))
    if not cases:
        raise ValueError("No completed inference cases")
    records = []
    for marker in cases:
        source_qc = json.loads((marker.parent / "stardist_qc.json").read_text())
        mask_path = Path(source_qc["nuclei_mask"])
        labels = tifffile.imread(mask_path)
        binary = (labels > 0).astype(np.float32)
        original = torch.from_numpy(binary)[None, None]
        coarse = F.interpolate(original, scale_factor=1 / 8,
                               mode="bilinear", align_corners=False)
        restored = F.interpolate(coarse, size=binary.shape,
                                 mode="bilinear", align_corners=False)[0, 0].numpy()
        score = NuclearAlignmentScore(binary.shape)
        score.add(restored, labels, 0, 0)
        ideal = score.report()
        records.append({"sample_id": marker.parent.name,
                        "shape_yx": list(binary.shape),
                        "actual_virtual_DAPI_status": source_qc["status"],
                        "actual_auc": source_qc["auc"],
                        "actual_boundary_contrast": source_qc["nuclear_boundary_contrast"],
                        "ideal_eightfold_coarse_status": ideal["status"],
                        "ideal_auc": ideal["auc"],
                        "ideal_boundary_contrast": ideal["nuclear_boundary_contrast"],
                        "ideal_spatial_status": ideal["spatial_shadow_status"]})
    report = {"status": "diagnostic_complete", "device": "CPU",
              "source": "https://github.com/prov-gigatime/GigaTIME/blob/main/scripts/db_train.py",
              "target_transform": "F.interpolate(binary_nuclear_mask, scale_factor=1/8, bilinear, align_corners=False), then bilinear resize to source shape",
              "completed_cases_examined": len(records),
              "actual_qc": dict(Counter(row["actual_virtual_DAPI_status"] for row in records)),
              "ideal_coarsened_qc": dict(Counter(row["ideal_eightfold_coarse_status"] for row in records)),
              "interpretation": "An optimistic self-mask control isolates target smoothing from model errors. Passing does not prove the GigaTIME model is correct; failing would show the fixed QC rejects even this idealized training target. The model was not rerun and thresholds were not changed.",
              "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--qc-module-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.inference, args.qc_module_dir, args.output)
    print(json.dumps({"cases": result["completed_cases_examined"],
                      "actual_qc": result["actual_qc"],
                      "ideal_coarsened_qc": result["ideal_coarsened_qc"]}))
