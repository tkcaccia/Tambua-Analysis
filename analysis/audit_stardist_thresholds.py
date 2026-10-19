#!/usr/bin/env python3
"""CPU-only StarDist threshold audit on immutable ORION windows.

Save alternate matched masks outside the source study. The GigaTIME image and
published ORION cell masks are never modified or used to fit StarDist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_orion_compartments import map_nuclei


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--prob", type=float, nargs="+", default=[.4, .55, .6924782541382084, .8])
    p.add_argument("--nms", type=float, default=.3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = StarDist2D.from_pretrained("2D_versatile_he")
    for case in args.cases:
        src = args.study / "cpu-prep" / case
        dst = args.output / case
        dst.mkdir(parents=True, exist_ok=True)
        he = tifffile.imread(src / "he.ome.tif")
        cells = tifffile.imread(src / "cells.ome.tif")
        prov = json.loads((src / "PROVENANCE.json").read_text())
        mpp = float(prov["mpp_x"])
        size = (round(he.shape[1] * mpp / .25), round(he.shape[0] * mpp / .25))
        rgb = np.asarray(Image.fromarray(he).resize(size, Image.Resampling.BICUBIC))
        t0 = time.monotonic()
        prob, dist = model.predict(normalize(rgb), n_tiles=(3, 3, 1), show_tile_progress=False)
        inference_s = time.monotonic() - t0
        for threshold in args.prob:
            t1 = time.monotonic()
            labels, _ = model._instances_from_prediction(
                rgb.shape[:2], prob, dist, prob_thresh=threshold, nms_thresh=args.nms)
            native = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(
                (he.shape[1], he.shape[0]), Image.Resampling.NEAREST))
            mapped, report = map_nuclei(native, cells)
            tag = f"p{threshold:.3f}".replace(".", "_")
            mask_path = dst / f"nuclei_{tag}.tif"
            tifffile.imwrite(mask_path, mapped, photometric="minisblack")
            report.update({
                "case": case, "prob_thresh": threshold, "nms_thresh": args.nms,
                "source_mpp": mpp, "stardist_grid_mpp": .25,
                "inference_seconds": inference_s,
                "postprocess_seconds": time.monotonic() - t1,
                "matched_cell_fraction": report["orion_cells_with_nucleus"] / report["orion_cells"],
                "detected_pixel_fraction_kept": report["nuclear_pixels_kept"] / max(1, report["nuclear_pixels_detected"]),
                "mask": str(mask_path),
            })
            (dst / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
