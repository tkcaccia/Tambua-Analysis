#!/usr/bin/env python3
"""Re-score persisted GigaTIME DAPI with alternate H&E StarDist masks (CPU)."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qc-code", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.qc_code.parent))
    spec = importlib.util.spec_from_file_location("gigatime_nuclear_qc", args.qc_code)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for case in args.cases:
        image = args.study / "gpu-output" / case / "gigatime" / "gigatime_probs.zarr"
        for tag in args.tags:
            folder = args.output / case
            metadata = json.loads((folder / f"{tag}.json").read_text())
            qc_path = folder / f"qc_{tag}.json"
            if not qc_path.exists():
                module.assess(str(image), str(folder / f"nuclei_{tag}.tif"), str(qc_path))
            qc = json.loads(qc_path.read_text())
            rows.append({
                "case": case, "split": "development" if int(case[3:]) <= 6 else "reserved",
                "threshold": metadata["prob_thresh"], "nms": metadata["nms_thresh"],
                "matched_cells": metadata["orion_cells_with_nucleus"],
                "orion_cells": metadata["orion_cells"],
                "matched_cell_fraction": metadata["matched_cell_fraction"],
                "detected_pixel_fraction_kept": metadata["detected_pixel_fraction_kept"],
                "nuclei_detected": metadata["stardist_nuclei"],
                "dapi_auc": qc.get("auc"),
                "dapi_dice": qc.get("prevalence_matched_dice"),
                "qc_status": qc["status"],
            })
            print(case, tag, round(qc.get("auc") or 0, 3), qc["status"], flush=True)
    with (args.output / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for split in ("development", "reserved"):
        cohort = [r for r in rows if r["split"] == split]
        for threshold in sorted({r["threshold"] for r in cohort}):
            group = [r for r in cohort if r["threshold"] == threshold]
            summary[f"{split}:{threshold:.6f}"] = {
                "cases": len(group),
                "mean_matched_cell_fraction": statistics.mean(r["matched_cell_fraction"] for r in group),
                "mean_dapi_auc": statistics.mean(r["dapi_auc"] for r in group),
                "qc_passes": sum(r["qc_status"] == "pass" for r in group),
            }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
