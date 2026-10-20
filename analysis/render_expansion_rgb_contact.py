#!/usr/bin/env python3
"""Render all CPU-prepared scanner-RGB MoNuSAC ROIs for visual preflight.

This checks input presentation, not GigaTIME output or post-inference QC.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


def render(prepared: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("RGB preflight must explicitly hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    prep = json.loads((prepared / "PREP_COMPLETE.json").read_text())
    rows = prep["records"]
    if prep["cases"] != 93 or len(rows) != 93 or not prep["all_scanner_rgb"]:
        raise ValueError("Frozen 93-case scanner-RGB preparation changed")
    from collections import Counter
    if Counter(row["split"] for row in rows) != {"train": 59, "test": 34}:
        raise ValueError("Prepared train/test split changed")
    output.mkdir(parents=True)
    cols, thumb_w, thumb_h, label_h = 5, 260, 260, 52
    page_size = 25
    pages = []
    for page_index, offset in enumerate(range(0, len(rows), page_size), start=1):
        batch = rows[offset:offset + page_size]
        canvas = Image.new("RGB", (cols * thumb_w,
                                   ((len(batch) + cols - 1) // cols) * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(batch):
            path = prepared / "samples" / row["sample_id"] / "he.ome.tif"
            with Image.open(path) as source:
                if source.mode != "RGB" or source.size != (row["width"], row["height"]):
                    raise ValueError(f"RGB mode or size mismatch: {row['sample_id']}")
                image = source.copy()
            image.thumbnail((thumb_w - 10, thumb_h - 10),
                            getattr(getattr(Image, "Resampling", Image), "LANCZOS"))
            x = (index % cols) * thumb_w
            y = (index // cols) * (thumb_h + label_h)
            canvas.paste(image, (x + (thumb_w - image.width)//2,
                                 y + (thumb_h - image.height)//2))
            draw.text((x+4, y+thumb_h+2), row["sample_id"][:30], fill="black")
            draw.text((x+4, y+thumb_h+19),
                      f"{row['organ']} | {row['split']} | {row['mpp']:.4g} um/px",
                      fill="black")
        name = f"rgb-preflight-{page_index:02d}.webp"
        canvas.save(output / name, "WEBP", quality=90, method=6)
        pages.append({"file": name, "sample_ids": [row["sample_id"] for row in batch]})
    result = {"status": "complete", "scope": "Source RGB visual preflight only; no GigaTIME output",
              "cases": len(rows), "pages": pages,
              "source_color_reference": "scanner_svs_rgb",
              "limit": "Visual review of this input montage cannot validate GigaTIME marker fidelity or shadows."}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps({key: value for key, value in render(args.prepared, args.output).items()
                      if key in ("status", "cases", "source_color_reference")}))
