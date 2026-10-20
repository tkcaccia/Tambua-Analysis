#!/usr/bin/env python3
"""Create one 47-image visual-audit contact sheet per GigaTIME channel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def build(root: Path) -> dict:
    source = root / "manifest.json"
    manifest = json.loads(source.read_text())
    if manifest.get("cases") != 47 or len(manifest.get("channel_names", [])) != 23 or \
            len(manifest.get("records", [])) != 47:
        raise ValueError("Incomplete frozen marker gallery")
    outputs = []
    for channel_index, name in enumerate(manifest["channel_names"]):
        path = root / f"contact-{channel_index:02d}.webp"
        if path.exists():
            raise FileExistsError(path)
        sheet = Image.new("RGB", (6 * 164, 8 * 132), "white")
        paint = ImageDraw.Draw(sheet)
        for index, row in enumerate(manifest["records"]):
            sample = row["sample_id"]
            with Image.open(root / f"{sample}--{channel_index:02d}.webp") as opened:
                image = opened.convert("RGB")
            image.thumbnail((158, 104), Image.Resampling.LANCZOS)
            x, y = index % 6 * 164, index // 6 * 132
            sheet.paste(image, (x, y + 22))
            paint.text((x + 2, y + 2), f"{index+1:02d}{row['split'][0]} {sample[:12]}",
                       fill="black")
        sheet.save(path, "WEBP", quality=85, method=4)
        outputs.append({"channel": name, "path": path.name})
    result = {"status": "complete", "cases_per_sheet": 47,
              "channels": outputs,
              "marker_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "layout": "Six columns; row-major case order follows manifest.records. Label: index, split initial, first 12 sample-ID characters. Contact sheets are visual previews, not quantitative expression maps."}
    staged = root / "CONTACTS_COMPLETE.json.tmp"
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(root / "CONTACTS_COMPLETE.json")
    return {"sheets": len(outputs), "cases_per_sheet": 47}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root)))
