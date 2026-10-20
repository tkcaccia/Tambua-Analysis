#!/usr/bin/env python3
"""Read-only, CPU-only spatial-channel diagnostic for frozen QC-pass fields.

Correlation is measured within independently detected nuclear pixels and is
not a measure of protein accuracy or a reason to override any QC gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import tifffile
import zarr


def inspect(inference: Path, sample: str, max_pixels: int = 20000) -> dict:
    case = inference / sample
    complete = json.loads((case / "CASE_COMPLETE.json").read_text())
    if complete["sample_id"] != sample or complete["nuclear_qc"]["status"] != "pass":
        raise ValueError(f"Case identity or nuclear QC failed: {sample}")
    store = zarr.open(str(case / "gigatime_output" / "gigatime_probs.zarr"), mode="r")
    array = store["0"]
    channels = list(store.attrs["gigatime"]["store_channels"])
    if array.shape[0] != 23 or len(channels) != 23 or channels[0] != "DAPI":
        raise ValueError(f"Unexpected native channel order: {sample}")
    labels = tifffile.imread(case / "stardist_labels.tif")
    yy = np.minimum(labels.shape[0] - 1, np.floor((np.arange(array.shape[1]) + .5)
                    * labels.shape[0] / array.shape[1]).astype(int))
    xx = np.minimum(labels.shape[1] - 1, np.floor((np.arange(array.shape[2]) + .5)
                    * labels.shape[1] / array.shape[2]).astype(int))
    nuclear = labels[yy[:, None], xx[None, :]] > 0
    indices = np.flatnonzero(nuclear.ravel())
    if len(indices) < 100:
        raise ValueError(f"Insufficient independent nuclear pixels: {sample}")
    selected = indices[np.linspace(0, len(indices) - 1,
                                   min(max_pixels, len(indices)), dtype=int)]
    values = np.stack([np.asarray(array[i]).ravel()[selected]
                       for i in range(len(channels))], axis=1)
    coefficient = spearmanr(values, axis=0).statistic
    if coefficient.shape != (23, 23) or not np.isfinite(coefficient).all():
        raise ValueError(f"Nonfinite or flat channel rank correlation: {sample}")
    upper = coefficient[np.triu_indices(23, k=1)]
    return {
        "sample_id": sample,
        "patient_id": complete.get("patient_id"),
        "organ": complete.get("organ"),
        "model_mpp_um": complete["model_mpp"],
        "nuclear_pixels_total": int(len(indices)),
        "nuclear_pixels_sampled": int(len(selected)),
        "channel_names": channels,
        "median_pairwise_spearman": float(np.median(upper)),
        "pair_fraction_spearman_gt_0_8": float(np.mean(upper > .8)),
        "dapi_to_other_spearman": {name: float(coefficient[0, i])
                                    for i, name in enumerate(channels) if i},
        "channel_pair_spearman": [
            {"first": channels[i], "second": channels[j],
             "rho": float(coefficient[i, j])}
            for i in range(23) for j in range(i + 1, 23)
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This diagnostic must hide GPUs")
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [inspect(args.inference, sample) for sample in args.samples]
    report = {
        "status": "diagnostic_only",
        "device": "CPU",
        "source_inference": str(args.inference),
        "sampling": "Deterministic uniform subsample of independent StarDist nuclear pixels",
        "records": records,
        "limit": "Cross-channel rank correlation measures spatial redundancy, not measured protein agreement or classifier accuracy. No QC status or output is altered.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({r["sample_id"]: r["median_pairwise_spearman"]
                      for r in records}, indent=2))


if __name__ == "__main__":
    main()
