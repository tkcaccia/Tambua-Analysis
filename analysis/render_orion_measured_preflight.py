#!/usr/bin/env python3
"""Render registered raw ORION fluorescence pixels for CPU-only source review.

These are measured images, not GigaTIME predictions or per-cell painted maps.
The display transform is per channel and is excluded from numeric scoring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile


MARKERS = ("Hoechst", "CD68", "CD4", "CD8a", "CD20", "PD-L1",
           "CD3e", "PD-1", "Ki67", "Pan-CK")


def contrast(array: np.ndarray) -> Image.Image:
    image = np.asarray(array, dtype=np.float32)
    scale = max(1.0, float(np.quantile(image, .95)))
    shifted = np.arcsinh(image / scale)
    lo, hi = np.quantile(shifted, [.01, .997])
    if hi <= lo:
        raise ValueError("Measured marker has no display contrast")
    rendered = np.rint(np.clip((shifted - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rendered).convert("RGB")


def render(study: Path, measured_root: Path, patient: str, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Measured-pixel preflight must explicitly hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    prepared = study / "cpu-prep" / patient
    source = json.loads((measured_root / patient / "MEASURED_PIXEL_SOURCE.json").read_text())
    provenance = json.loads((prepared / "PROVENANCE.json").read_text())
    if source["patient"] != patient or any(source[k] != provenance[k]
                                            for k in ("x", "y", "size_px", "mpp_x", "mpp_y")):
        raise ValueError("Measured window and registered H&E differ")
    if tuple(row["marker"] for row in source["channels"]) != MARKERS:
        raise ValueError("Ten-channel measured panel changed")
    with Image.open(prepared / "he.ome.tif") as handle:
        he = handle.convert("RGB")
    if he.size != (provenance["size_px"], provenance["size_px"]):
        raise ValueError("H&E window shape changed")
    width = 360
    panel = Image.new("RGB", (width * 4, (width + 30) * 3), "white")
    painter = ImageDraw.Draw(panel)
    images = [("H&E", he)]
    for marker in MARKERS:
        values = tifffile.imread(measured_root / patient / f"{marker}.tif")
        if values.shape != he.size[::-1] or values.dtype != np.uint16:
            raise ValueError(f"Invalid raw measured image: {marker}")
        images.append((marker, contrast(values)))
    for index, (name, image) in enumerate(images):
        image.thumbnail((width, width), getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
        x, y = (index % 4) * width, (index // 4) * (width + 30)
        panel.paste(image, (x + (width - image.width)//2, y + (width - image.height)//2))
        painter.text((x + 8, y + width + 5), name, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, "WEBP", quality=86, method=6)
    return {"patient": patient, "markers": MARKERS,
            "source_window_xy": [source["x"], source["y"]],
            "source_mpp": source["mpp_x"], "output": str(output),
            "scope": "Raw measured ORION fluorescence and registered H&E; no virtual-marker inference"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(render(args.study, args.measured, args.patient, args.output)))
