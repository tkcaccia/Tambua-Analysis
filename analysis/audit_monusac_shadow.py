#!/usr/bin/env python3
"""Audit every saved MoNuSAC virtual-DAPI preview against expert nuclei.

The historical outputs persisted JPEG marker previews but no native score
store. These measurements are diagnostic only; they cannot certify marker
quantification or replace a rerun with the native-store CellPhenotyper QC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    if pos.size < 100 or neg.size < 100:
        return None
    positive = np.bincount(pos, minlength=256).astype(np.int64)
    negative = np.bincount(neg, minlength=256).astype(np.int64)
    below = np.cumsum(negative) - negative
    return float(np.sum(positive * (below + 0.5 * negative)) / (pos.size * neg.size))


def assess_sample(source: Path, sample_id: str) -> dict:
    prepared = source / "prepared" / sample_id
    generated = source / "output" / sample_id
    with Image.open(prepared / "expert_nuclei.ome.tif") as image:
        expert = np.asarray(image, dtype=np.int32)
    with Image.open(generated / "gigatime_output/jpg_image/DAPI.jpg") as image:
        dapi = np.asarray(image.convert("L"), dtype=np.uint8)
    if not expert.any() or min(dapi.shape) < 300:
        raise ValueError(f"Insufficient expert nuclei or small image: {sample_id}")
    if expert.shape != dapi.shape:
        expert = np.asarray(Image.fromarray(expert).resize(
            (dapi.shape[1], dapi.shape[0]), Image.Resampling.NEAREST))
    nucleus = expert > 0
    nearby = binary_dilation(nucleus, iterations=8) & ~nucleus
    h, w = dapi.shape
    local = []
    for row in range(3):
        for col in range(3):
            ys = slice(row * h // 3, (row + 1) * h // 3)
            xs = slice(col * w // 3, (col + 1) * w // 3)
            p, n = dapi[ys, xs][nucleus[ys, xs]], dapi[ys, xs][nearby[ys, xs]]
            local.append({"row": row, "column": col, "auc": auc(p, n),
                          "nuclear_pixels": int(p.size), "nearby_pixels": int(n.size)})
    edge = np.zeros((h, w), dtype=bool)
    edge[:24] = edge[-24:] = True
    edge[:, :24] = edge[:, -24:] = True
    edge_n, core_n = dapi[nucleus & edge], dapi[nucleus & ~edge]
    return {
        "sample_id": sample_id, "shape_yx": [h, w],
        "expert_nuclei": int(expert.max()),
        "jpeg_nearby_nucleus_auc": auc(dapi[nucleus], dapi[nearby]),
        "jpeg_local_auc": local,
        "eligible_local_regions": sum(x["auc"] is not None for x in local),
        "local_regions_below_0_60": sum(x["auc"] is not None and x["auc"] < .60 for x in local),
        "edge_to_core_nuclear_mean_ratio": (
            float(edge_n.mean() / core_n.mean())
            if edge_n.size >= 100 and core_n.size >= 500 and core_n.mean() > 0 else None),
    }


def audit(source: Path) -> dict:
    cohort = json.loads((source / "input/COHORT.json").read_text())
    records = cohort["records"]
    if len(records) != 120:
        raise ValueError("Expected the frozen 120-image cohort")
    cases = []
    for record in records:
        case = assess_sample(source, record["sample_id"])
        case.update(split=record["split"], organ=record["organ"])
        cases.append(case)
    return {
        "scope": "historical_JPEG_preview_diagnostic_not_native_marker_QC",
        "cohort_images": len(records), "audited_images": len(cases),
        "images_below_global_0_70": sum(x["jpeg_nearby_nucleus_auc"] is None or
                                            x["jpeg_nearby_nucleus_auc"] < .70 for x in cases),
        "images_with_local_below_0_60": sum(x["local_regions_below_0_60"] > 0 for x in cases),
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
