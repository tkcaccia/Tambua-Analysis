#!/usr/bin/env python3
"""Render every prepared NuCLS RGB field and typed outline for visual audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import tifffile


COLORS = {"tumor_epithelial": (240, 40, 190),
          "lymphocyte": (0, 220, 255),
          "macrophage": (255, 100, 20),
          "plasma_cell": (240, 210, 0)}
SIDE = 180
PER_PAGE = 12


def fitted(image: np.ndarray) -> Image.Image:
    item = Image.fromarray(image, "RGB")
    item.thumbnail((SIDE-6, SIDE-6), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (SIDE, SIDE), "white")
    tile.paste(item, ((SIDE-item.width)//2, (SIDE-item.height)//2))
    return tile


def pair(folder: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    rgb = tifffile.imread(folder/"he.ome.tif")
    labels = tifffile.imread(folder/"expert_target_nuclei.ome.tif")
    with (folder/"expert_types.csv").open(newline="") as stream:
        table = {int(r["label_id"]):r["cell_type"] for r in csv.DictReader(stream)}
    if rgb.shape[:2] != labels.shape or rgb.dtype != np.uint8 or \
            set(np.unique(labels).astype(int)) - {0} != set(table):
        raise ValueError(f"Prepared RGB and expert target mask disagree: {folder}")
    overlay = rgb.copy()
    counts = {name: 0 for name in COLORS}
    for label_id, name in table.items():
        if name not in COLORS:
            raise ValueError(f"Unknown NuCLS type: {name}")
        mask = labels == label_id
        edge = mask & ~binary_erosion(mask, iterations=1)
        overlay[edge] = COLORS[name]
        counts[name] += 1
    return rgb, overlay, counts


def run(prepared: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Source-contact renderer must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((prepared/"PREP_COMPLETE.json").read_text())
    if source["status"] != "source_prepared_no_inference" or \
            source["cases"] != 192:
        raise ValueError("NuCLS source cohort not prepared")
    records = sorted(source["records"], key=lambda r:(r["split"],r["fov_id"]))
    output.mkdir(parents=True)
    pages = []
    for start in range(0, len(records), PER_PAGE):
        batch = records[start:start+PER_PAGE]
        canvas = Image.new("RGB", (6*SIDE, 4*(SIDE+45)+38), "white")
        pen = ImageDraw.Draw(canvas)
        pen.text((6,8), "NuCLS native H&E RGB | typed expert polyline outlines: "
                 "orange macrophage, cyan lymphocyte, magenta tumor epithelial, "
                 "yellow plasma", fill="black")
        for index, record in enumerate(batch):
            fov = record["fov_id"]
            rgb, overlay, counts = pair(prepared/fov)
            col, row = index%3, index//3
            x, y = col*2*SIDE, 38+row*(SIDE+45)
            pen.text((x+4,y+2), f"{record['patient_id']} {record['split']} "
                     f"M{counts['macrophage']} L{counts['lymphocyte']} "
                     f"E{counts['tumor_epithelial']} P{counts['plasma_cell']}",
                     fill="black")
            canvas.paste(fitted(rgb), (x,y+38))
            canvas.paste(fitted(overlay), (x+SIDE,y+38))
        filename = f"source-review-{start//PER_PAGE+1:02d}.webp"
        canvas.save(output/filename,"WEBP",quality=88,method=4)
        pages.append({"file":filename,"fov_ids":[r["fov_id"] for r in batch]})
    report={"status":"pending_manual_review","images":len(records),
            "pages":pages,"panels":["native scanner RGB", "typed polyline overlay"],
            "source_mpp_um":.2,"target_model_mpp_um":.25,
            "limit":"Pre-GigaTIME source review only. It does not validate virtual markers, StarDist, or classification."}
    (output/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    x=run(a.prepared,a.output)
    print(json.dumps({"images":x["images"],"pages":len(x["pages"])}))
