#!/usr/bin/env python3
"""CPU-only contact sheet of released HEMIT input and target channels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tifffile


def display_channel(channel: np.ndarray) -> Image.Image:
    values = np.asarray(channel, dtype=np.float32)
    low, high = np.percentile(values, (1, 99.5))
    if high <= low:
        high = low + 1
    scaled = np.clip((values - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    return Image.fromarray(scaled).convert("RGB")


def display_rgb(rgb: np.ndarray) -> Image.Image:
    if rgb.dtype == np.uint8:
        return Image.fromarray(rgb, "RGB")
    bands = [np.asarray(display_channel(rgb[..., index]).convert("L")) for index in range(3)]
    return Image.fromarray(np.stack(bands, axis=-1), "RGB")


def render(root: Path, ids: list[str], out: Path) -> None:
    cell = 256
    columns = ("H&E RGB", "Target composite", "Target band 0", "Target band 1", "Target band 2")
    sheet = Image.new("RGB", (len(columns) * cell, len(ids) * (cell + 48) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(columns):
        draw.text((column * cell + 6, 6), title, fill="black")
    for row, ident in enumerate(ids):
        he = tifffile.imread(root / "test" / "input" / ident)
        target = tifffile.imread(root / "test" / "label" / ident)
        if he.shape != (1024, 1024, 3) or target.shape != (1024, 1024, 3):
            raise ValueError(f"Unexpected HEMIT pair dimensions: {ident}")
        images = [display_rgb(he), display_rgb(target)]
        images += [display_channel(target[..., index]) for index in range(3)]
        top = 28 + row * (cell + 48)
        draw.text((6, top + 5), ident, fill="black")
        for column, image in enumerate(images):
            thumb = ImageOps.contain(image, (cell, cell), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (column * cell, top + 32))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "WEBP", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ids", type=Path,
                        default=Path(__file__).with_name("hemit_test_no_overlap_no_empty_ids.txt"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("--count must be between 1 and 20")
    ids = [name.strip() for name in args.ids.read_text().splitlines() if name.strip()]
    render(args.root, ids[:args.count], args.output)


if __name__ == "__main__":
    main()
