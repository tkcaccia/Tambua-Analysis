#!/usr/bin/env python3
"""Render train-derived fixed-shift QC candidates for manual spatial review.

The renderer is diagnostic. The original files and official QC are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, shift as nd_shift
import tifffile
import zarr

from render_all_monusac_markers import CHANNELS, display


PANELS = ("Scanner RGB", "DAPI original", "DAPI shifted",
          "Shifted DAPI + StarDist", "CK shifted", "CD68 shifted")
SIZE = 250


def tile(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((SIZE-8, SIZE-8), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (SIZE, SIZE), "white")
    out.paste(image, ((SIZE-image.width)//2, (SIZE-image.height)//2))
    return out


def moved(channel: np.ndarray) -> np.ndarray:
    return np.clip(nd_shift(channel.astype(np.float32), (-3, 3),
                            order=1, mode="constant", cval=0,
                            prefilter=False), 0, 65535).astype(np.uint16)


def render(prepared: Path, inference: Path, score_path: Path,
           output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    score = json.loads(score_path.read_text())
    selected = [r for r in score["records"] if r["frozen_status"] == "fail"
                and r["fixed_shift_status"] == "pass"]
    if score["train_selected_shift_model_px"] != [-3, 3] or not selected:
        raise ValueError("Missing frozen shift or new QC candidates")
    source = {r["sample_id"]: r for r in
              json.loads((prepared / "PREP_COMPLETE.json").read_text())["records"]}
    output.mkdir(parents=True)
    pages = []
    for start in range(0, len(selected), 4):
        batch = selected[start:start+4]
        canvas = Image.new("RGB", (SIZE*len(PANELS),
                                   44+len(batch)*(SIZE+42)), "white")
        pen = ImageDraw.Draw(canvas)
        for i, name in enumerate(PANELS):
            pen.text((i*SIZE+5, 12), name, fill="black")
        for row_no, row in enumerate(batch):
            sample = row["sample_id"]
            case = inference / sample
            rgb_path = prepared / "samples" / sample / "he.ome.tif"
            saved = json.loads((case / "CASE_COMPLETE.json").read_text())
            if saved["source_rgb_sha256"] != hashlib.sha256(rgb_path.read_bytes()).hexdigest() or \
                    saved["sample_id"] != sample or source[sample]["split"] != row["split"]:
                raise ValueError(f"RGB/source identity changed: {sample}")
            with Image.open(rgb_path) as im:
                rgb = im.convert("RGB")
            store = zarr.open(str(case / "gigatime_output" /
                                  "gigatime_probs.zarr"), mode="r")
            array = store["0"]
            if tuple(store.attrs["gigatime"]["store_channels"]) != CHANNELS:
                raise ValueError(f"Channel order changed: {sample}")
            raw_dapi = np.asarray(array[CHANNELS.index("DAPI")])
            shifted_dapi = moved(raw_dapi)
            mask = tifffile.imread(case / "stardist_labels.tif")
            yy = np.minimum(mask.shape[0]-1, np.floor(
                (np.arange(raw_dapi.shape[0])+.5)*mask.shape[0]/raw_dapi.shape[0]).astype(int))
            xx = np.minimum(mask.shape[1]-1, np.floor(
                (np.arange(raw_dapi.shape[1])+.5)*mask.shape[1]/raw_dapi.shape[1]).astype(int))
            nuclear = mask[yy[:, None], xx[None, :]] > 0
            outline = nuclear & ~binary_erosion(nuclear, iterations=1)
            gray = display(shifted_dapi)
            overlay = np.stack([gray//2]*3, axis=-1)
            overlay[outline] = [0, 255, 80]
            panels = (rgb, Image.fromarray(display(raw_dapi)),
                      Image.fromarray(gray), Image.fromarray(overlay),
                      Image.fromarray(display(moved(np.asarray(array[CHANNELS.index("CK")])))),
                      Image.fromarray(display(moved(np.asarray(array[CHANNELS.index("CD68")])))))
            top = 44+row_no*(SIZE+42)
            pen.text((5, top+3), f"{sample} | {row['organ']} {row['split']} | "
                     f"AUC {row['frozen_auc']:.3f} to {row['fixed_shift_auc']:.3f}",
                     fill="black")
            for col, image in enumerate(panels):
                canvas.paste(tile(image), (col*SIZE, top+38))
        file = f"fixed-shift-review-{start//4+1:02d}.webp"
        canvas.save(output/file, "WEBP", quality=88, method=4)
        pages.append({"file": file, "sample_ids": [r["sample_id"] for r in batch]})
    report = {"status": "pending_manual_review", "selected_cases": len(selected),
              "fixed_shift_model_px": [-3, 3], "pages": pages,
              "limit": "A diagnostic QC replay, not a validated marker correction. Every candidate requires visual review; test-set labels were not used to select the shift."}
    (output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--score", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render(a.prepared, a.inference, a.score, a.output)))
