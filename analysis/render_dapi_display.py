#!/usr/bin/env python3
"""Render a mask-independent DAPI preview and optional StarDist contour overlay.

This is a visualization step only. Pass the unmodified native GigaTIME channel
to CellPhenotyper's nuclear QC and marker quantification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_erosion


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(linear_png: Path, labels_tif: Path, output: Path, code: Path) -> dict:
    sys.path.insert(0, str(code / "bin"))
    from gigatime_display import tone_map_dapi_preview

    linear = np.asarray(Image.open(linear_png).convert("L"))
    enhanced, transform = tone_map_dapi_preview(linear)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced).save(output / "dapi_display.png")

    labels_native = tifffile.imread(labels_tif)
    labels = np.asarray(
        Image.fromarray(labels_native.astype(np.int32)).resize(
            (linear.shape[1], linear.shape[0]), Image.Resampling.NEAREST
        )
    )
    boundary = (labels > 0) & ~binary_erosion(labels > 0)
    overlay = np.repeat(enhanced[..., None], 3, axis=2)
    overlay[boundary] = (255, 35, 70)
    Image.fromarray(overlay).save(output / "dapi_display_stardist_overlay.png")
    report = {
        "linear_png_sha256": sha256(linear_png),
        "stardist_labels_sha256": sha256(labels_tif),
        "linear_shape_yx": list(linear.shape),
        "stardist_native_shape_yx": list(labels_native.shape),
        "stardist_object_count": int(np.unique(labels_native[labels_native > 0]).size),
        "overlay_color_rgb": [255, 35, 70],
        "transform": transform,
        "qc_note": "Display transform did not use StarDist, expert nuclei or QC metrics; score raw native GigaTIME output separately.",
    }
    (output / "dapi_display_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-png", type=Path, required=True)
    ap.add_argument("--stardist-labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cellphenotyper-code", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(render(args.linear_png, args.stardist_labels, args.output, args.cellphenotyper_code), indent=2))
