#!/usr/bin/env python3
"""Compare spatial separation of GigaTIME channels on saved QC-passing fields.

CPU-only, read-only diagnostic. High marker correlation can reflect shared
anatomy; this score is not a marker-validity gate by itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import tifffile
import zarr

from render_all_monusac_markers import CHANNELS


PAIRS = (("DAPI", "TRITC"), ("DAPI", "Cy5"), ("DAPI", "CK"),
         ("CK", "CD68"), ("CK", "CD3"), ("CD68", "CD3"))


def one(sample: str, rgb_path: Path, store_path: Path, stride: int) -> dict:
    with tifffile.TiffFile(rgb_path) as tif:
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB:
            raise ValueError(f"Source must be tagged RGB: {sample}")
        rgb = tif.asarray()
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Source must be uint8 scanner RGB: {sample}")
    store = zarr.open(str(store_path), mode="r")
    data = store["0"]
    meta = store.attrs["gigatime"]
    if data.shape[0] != len(CHANNELS) or data.dtype != np.uint16 or \
            tuple(meta["store_channels"]) != CHANNELS or \
            abs(float(meta["effective_mpp"]) - .25) > 1e-6:
        raise ValueError(f"Spatial-marker identity or calibration changed: {sample}")
    h, w = data.shape[1:]
    resized = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
    pixels = resized[::stride, ::stride].reshape(-1, 3)
    tissue = (pixels.mean(axis=1) < 240) & (np.ptp(pixels, axis=1) > 12)
    if int(tissue.sum()) < 100:
        raise ValueError(f"Insufficient tissue for marker comparison: {sample}")
    ranks = []
    for index in range(len(CHANNELS)):
        values = np.asarray(data[index, ::stride, ::stride]).reshape(-1)[tissue]
        if np.ptp(values) == 0:
            raise ValueError(f"Constant marker: {sample}/{CHANNELS[index]}")
        ranks.append(rankdata(values, method="average"))
    corr = np.corrcoef(np.stack(ranks))
    if not np.isfinite(corr).all():
        raise ValueError(f"Nonfinite channel correlation: {sample}")
    upper = np.abs(corr[np.triu_indices(len(CHANNELS), 1)])
    return {"sample_id": sample, "sampled_tissue_pixels": int(tissue.sum()),
            "median_absolute_pairwise_spatial_spearman": float(np.median(upper)),
            "fraction_channel_pairs_absolute_rho_gt_0_90": float(np.mean(upper > .90)),
            "selected_pair_spatial_spearman": {
                f"{a}__{b}": float(corr[CHANNELS.index(a), CHANNELS.index(b)])
                for a, b in PAIRS}}


def run(prior_summary: Path, prior_rgb_root: Path, prior_output_root: Path,
        expansion_prepared: Path, expansion_output: Path, output: Path,
        stride: int = 8) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Specificity audit must hide GPUs")
    if output.exists() or stride < 1:
        raise ValueError("Output exists or stride invalid")
    qc = json.loads(prior_summary.read_text())
    prior = [r["sample_id"] for r in qc["records"]
             if r["stardist_nuclear_qc"]["status"] == "pass"]
    if len(prior) != 47:
        raise ValueError("Frozen prior QC cohort must have 47 passes")
    old = [one(sample, prior_rgb_root / sample / "he.ome.tif",
               prior_output_root / "output" / sample / "gigatime_output" / "gigatime_probs.zarr",
               stride) for sample in prior]
    new = []
    for case in sorted(expansion_output.glob("*/CASE_COMPLETE.json")):
        source = json.loads(case.read_text())
        nuclear = json.loads((case.parent / "stardist_qc.json").read_text())
        if nuclear["status"] != "pass" or source["seam_qc"] != "pass":
            continue
        sample = case.parent.name
        new.append(one(sample, expansion_prepared / "samples" / sample / "he.ome.tif",
                       case.parent / "gigatime_output" / "gigatime_probs.zarr", stride))
    report = {"status": "diagnostic_only", "device": "CPU", "stride": stride,
              "prior_qc_pass_fields": len(old), "expansion_qc_pass_fields": len(new),
              "method": "Rank native uint16 values of each spatial channel at every eighth model-grid pixel restricted to scanner-RGB tissue; compare all 23 channels. No display stretch or cell labels are used.",
              "interpretation_limit": "High spatial correlation may reflect common anatomy; this diagnostic alone cannot reject or validate a protein channel. The contrast between historical and expansion fields can reveal gross channel collapse.",
              "prior_median_absolute_pairwise_spatial_spearman": float(np.median(
                  [r["median_absolute_pairwise_spatial_spearman"] for r in old])),
              "prior_records": old, "expansion_records": new}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--prior-rgb-root", type=Path, required=True)
    parser.add_argument("--prior-output-root", type=Path, required=True)
    parser.add_argument("--expansion-prepared", type=Path, required=True)
    parser.add_argument("--expansion-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prior_summary, args.prior_rgb_root, args.prior_output_root,
                 args.expansion_prepared, args.expansion_output, args.output)
    print(json.dumps({"prior": result["prior_qc_pass_fields"],
                      "expansion": result["expansion_qc_pass_fields"],
                      "prior_median_abs_rho": result["prior_median_absolute_pairwise_spatial_spearman"]}))
