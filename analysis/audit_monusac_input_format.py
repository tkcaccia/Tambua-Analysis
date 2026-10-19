#!/usr/bin/env python3
"""Audit MoNuSAC staging against canonical scanner RGB, not just its source file."""

from __future__ import annotations

import argparse
import io
import json
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
import tifffile


def decode_svs_rgb(data: bytes) -> np.ndarray:
    """Match the preparation script's OpenSlide level-zero color conversion."""
    with tempfile.NamedTemporaryFile(suffix=".svs") as temporary:
        temporary.write(data)
        temporary.flush()
        with openslide.OpenSlide(temporary.name) as slide:
            return np.asarray(slide.read_region((0, 0), 0, slide.dimensions).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--staged-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    members = {}
    archives = [zipfile.ZipFile(path) for path in args.archive]
    for archive in archives:
        for name in archive.namelist():
            if name.lower().endswith((".tif", ".svs")):
                members[(Path(name).stem, Path(name).suffix.lower())] = (archive, name)

    rows = []
    for staged in sorted(Path(args.staged_root).glob("*/he.ome.tif")):
        sample_id = staged.parent.name
        provenance = json.loads((staged.parent / "provenance.json").read_text())
        source_member = provenance["source_image_member"]
        source_archive = next(archive for archive in archives if source_member in archive.namelist())
        source_bytes = source_archive.read(source_member)
        with tifffile.TiffFile(io.BytesIO(source_bytes)) as source_tf:
            page = source_tf.pages[0]
            source_compression = int(page.compression)
            source_mode = str(page.photometric)
            source_bits = tuple(page.bitspersample) if isinstance(page.bitspersample, tuple) else [int(page.bitspersample)]
        if source_member.lower().endswith(".svs"):
            # Use the same decoder and color handling as cohort staging.
            source = decode_svs_rgb(source_bytes)
            alpha_values = []
            decoder = "OpenSlide_RGB"
        else:
            with Image.open(io.BytesIO(source_bytes)) as source_im:
                raw = np.asarray(source_im)
                alpha_values = np.unique(raw[:, :, 3]).tolist() if raw.ndim == 3 and raw.shape[-1] == 4 else []
                source = np.asarray(source_im.convert("RGB"))
            decoder = "Pillow_RGB"
        with Image.open(staged) as staged_im:
            staged_rgb = np.asarray(staged_im)
            staged_compression = int(staged_im.tag_v2.get(259, -1))
            description = str(staged_im.tag_v2.get(270, ""))
        scanner_archive, scanner_member = members[(sample_id, ".svs")]
        scanner = scanner_archive.read(scanner_member)
        with tifffile.TiffFile(io.BytesIO(scanner)) as scanner_tf:
            scanner_compression = int(scanner_tf.pages[0].compression)
        scanner_rgb = decode_svs_rgb(scanner)
        if scanner_rgb.shape != source.shape:
            raise ValueError(f"Scanner/source shape mismatch: {sample_id}")
        scanner_same = bool(np.array_equal(scanner_rgb, source))
        scanner_swap = bool(np.array_equal(scanner_rgb, source[..., ::-1]))
        if scanner_same:
            scanner_channel_order = "RGB"
        elif scanner_swap:
            scanner_channel_order = "BGR_mistagged_as_RGB"
        else:
            scanner_channel_order = "other_difference"
        mpp_match = re.search(rb"\bMPP\s*=\s*([0-9.]+)", scanner)
        mag_match = re.search(rb"\bAppMag\s*=\s*([0-9.]+)", scanner)
        ome_mpp = re.search(r'PhysicalSizeX="([0-9.]+)"', description)
        rows.append({
            "sample_id": sample_id,
            "source_mode": source_mode,
            "source_extension": Path(source_member).suffix.lower(),
            "source_decoder": decoder,
            "source_bits": source_bits,
            "source_compression_code": source_compression,
            "source_shape": list(source.shape),
            "source_alpha_values": alpha_values,
            "scanner_description_has_icc_profile": b"ICC Profile" in scanner,
            "scanner_compression_code": scanner_compression,
            "scanner_channel_order_of_staged_source": scanner_channel_order,
            "scanner_source_same_rgb": scanner_same,
            "scanner_source_red_blue_swap_exact": scanner_swap,
            "scanner_source_mae_rgb": float(np.abs(scanner_rgb.astype(np.int16) - source.astype(np.int16)).mean()),
            "scanner_source_mae_red_blue_swapped": float(np.abs(scanner_rgb.astype(np.int16) - source[..., ::-1].astype(np.int16)).mean()),
            "staged_compression_code": staged_compression,
            "staged_shape": list(staged_rgb.shape),
            "rgb_pixel_exact": bool(np.array_equal(source[:, :, :3], staged_rgb)),
            "rgb_mean_absolute_decoder_difference": float(np.abs(source[:, :, :3].astype(np.int16) - staged_rgb.astype(np.int16)).mean()),
            "scanner_mpp": float(mpp_match.group(1)) if mpp_match else None,
            "scanner_appmag": float(mag_match.group(1)) if mag_match else None,
            "staged_ome_mpp": float(ome_mpp.group(1)) if ome_mpp else None,
        })

    for archive in archives:
        archive.close()
    summary = {
        "n": len(rows),
        "source_modes": dict(Counter(row["source_mode"] for row in rows)),
        "source_extensions": dict(Counter(row["source_extension"] for row in rows)),
        "source_compression_codes": dict(Counter(row["source_compression_code"] for row in rows)),
        "source_lossy_jpeg2000_count": sum(row["source_compression_code"] == 33005 for row in rows),
        "scanner_compression_codes": dict(Counter(row["scanner_compression_code"] for row in rows)),
        "scanner_source_channel_order": dict(Counter(row["scanner_channel_order_of_staged_source"] for row in rows)),
        "staged_canonical_rgb_count": sum(row["scanner_source_same_rgb"] for row in rows),
        "staged_red_blue_correction_needed_count": sum(row["scanner_source_red_blue_swap_exact"] and not row["scanner_source_same_rgb"] for row in rows),
        "staged_compression_codes": dict(Counter(row["staged_compression_code"] for row in rows)),
        "scanner_appmag": dict(Counter(row["scanner_appmag"] for row in rows)),
        "all_rgb_pixel_exact": all(row["rgb_pixel_exact"] for row in rows),
        "rgb_pixel_exact_count": sum(row["rgb_pixel_exact"] for row in rows),
        "all_alpha_opaque": all(not row["source_alpha_values"] or row["source_alpha_values"] == [255] for row in rows),
        "all_ome_mpp_matches_scanner": all(row["staged_ome_mpp"] == row["scanner_mpp"] for row in rows),
        "mpp_range": [min(row["scanner_mpp"] for row in rows), max(row["scanner_mpp"] for row in rows)],
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
