#!/usr/bin/env python3
"""Map independent H&E StarDist nuclei onto ORION measured-cell IDs.

GigaTIME is never used in this segmentation step. Cytoplasm means the published
whole-cell region after subtraction of its matched nuclear pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def map_nuclei(nuclei: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, dict]:
    if nuclei.shape != cells.shape:
        raise ValueError("Nuclear and cell masks are on different grids")
    nuclear_ids = nuclei.astype(np.int64, copy=False)
    cell_ids = cells.astype(np.int64, copy=False)
    if nuclear_ids.min() < 0 or cell_ids.min() < 0:
        raise ValueError("Negative labels are invalid")
    both = (nuclear_ids > 0) & (cell_ids > 0)
    pairs = (nuclear_ids[both] << 32) | cell_ids[both]
    code, count = np.unique(pairs, return_counts=True)
    star_ids = (code >> 32).astype(np.int64)
    orion_ids = (code & ((1 << 32) - 1)).astype(np.int64)
    total = np.bincount(nuclear_ids.ravel(), minlength=int(nuclear_ids.max()) + 1)
    best = np.zeros(len(total), np.int64)
    best_count = np.zeros(len(total), np.int64)
    for star, orion, n in zip(star_ids, orion_ids, count):
        if n > best_count[star]:
            best[star], best_count[star] = orion, n
    accepted = (best_count >= np.maximum(4, np.ceil(total * .5)))
    best[~accepted] = 0
    mapped = np.where((cell_ids > 0) & (best[nuclear_ids] == cell_ids), cell_ids, 0).astype(np.uint32)
    matched_cells = np.unique(mapped[mapped > 0])
    report = {"stardist_nuclei": int(np.count_nonzero(total[1:])),
              "matched_nuclei": int(np.count_nonzero(accepted[1:])),
              "orion_cells_with_nucleus": int(len(matched_cells)),
              "orion_cells": int(len(np.unique(cell_ids)) - 1),
              "nuclear_pixels_kept": int(np.count_nonzero(mapped)),
              "nuclear_pixels_detected": int(np.count_nonzero(nuclear_ids))}
    return mapped, report


def prepare(window: Path, *, prob_thresh: float = .52,
            nms_thresh: float = .28) -> dict:
    provenance = json.loads((window / "PROVENANCE.json").read_text())
    source_mpp = float(provenance["mpp_x"])
    he = tifffile.imread(window / "he.ome.tif")
    cells = tifffile.imread(window / "cells.ome.tif")
    if he.shape[:2] != cells.shape or he.ndim != 3 or he.shape[2] != 3:
        raise ValueError("Invalid ORION H&E/mask pair")
    target_size = (round(he.shape[1] * source_mpp / .25),
                   round(he.shape[0] * source_mpp / .25))
    resized = np.asarray(Image.fromarray(he).resize(target_size, Image.Resampling.BICUBIC))
    model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, _ = model.predict_instances(normalize(resized), n_tiles=(3, 3, 1),
                                        prob_thresh=prob_thresh,
                                        nms_thresh=nms_thresh)
    at_source = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(
        (he.shape[1], he.shape[0]), Image.Resampling.NEAREST))
    matched, report = map_nuclei(at_source, cells)
    cytoplasm = np.where(matched == 0, cells, 0).astype(np.uint32)
    tifffile.imwrite(window / "nuclei_orion_ids.ome.tif", matched, ome=True,
                     metadata={"axes": "YX", "PhysicalSizeX": source_mpp,
                               "PhysicalSizeY": source_mpp})
    tifffile.imwrite(window / "cytoplasm_orion_ids.ome.tif", cytoplasm, ome=True,
                     metadata={"axes": "YX", "PhysicalSizeX": source_mpp,
                               "PhysicalSizeY": source_mpp})
    report.update({"patient": provenance["patient"], "source_mpp": source_mpp,
                   "stardist_grid_mpp": .25, "nuclei_source": "H&E StarDist2D 2D_versatile_he",
                   "stardist_prob_thresh": prob_thresh,
                   "stardist_nms_thresh": nms_thresh,
                   "matching": "majority overlap, at least 50% of nucleus and four pixels"})
    (window / "COMPARTMENT_QC.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("window", type=Path)
    ap.add_argument("--prob-thresh", type=float, default=.52,
                    help="Match CellPhenotyper's configured StarDist probability threshold")
    ap.add_argument("--nms-thresh", type=float, default=.28,
                    help="Match CellPhenotyper's configured StarDist NMS threshold")
    args = ap.parse_args()
    if not (0 < args.prob_thresh < 1 and 0 < args.nms_thresh < 1):
        ap.error("StarDist thresholds must lie strictly between 0 and 1")
    print(json.dumps(prepare(args.window, prob_thresh=args.prob_thresh,
                             nms_thresh=args.nms_thresh), indent=2))


if __name__ == "__main__":
    main()
