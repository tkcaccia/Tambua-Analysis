#!/usr/bin/env python3
"""Render one validated CRC04 pilot's native spatial GigaTIME marker images.

Measured panels are registered raw ORION fluorescence, never cell-painted
abundance. This CPU-only diagnostic is not a cross-patient validation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
from scipy.stats import spearmanr
import tifffile
import zarr

from render_orion_pixel_dapi import display
from validate_orion_all_marker_pilot import CHANNELS


PAIRS = {
    "DAPI": ("Hoechst", "nuclear-dye family"),
    "CD68": ("CD68", "exact"),
    "CD4": ("CD4", "exact"),
    "CD8": ("CD8a", "CD8 alpha target"),
    "CD20": ("CD20", "exact"),
    "PD-L1": ("PD-L1", "exact"),
    "CD3": ("CD3e", "CD3 epsilon complex"),
    "PD-1": ("PD-1", "exact"),
    "Ki67": ("Ki67", "exact"),
    "CK": ("Pan-CK", "cytokeratin family"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source_checks(input_case: Path, measured_root: Path, pilot_root: Path) -> tuple[Image.Image, dict]:
    technical = json.loads((pilot_root / "CRC04" / "TECHNICAL_COMPLETE.json").read_text())
    done = json.loads((pilot_root / "PILOT_COMPLETE.json").read_text())
    prep = json.loads((input_case / "PROVENANCE.json").read_text())
    raw = json.loads((measured_root / "CRC04" / "MEASURED_PIXEL_SOURCE.json").read_text())
    if done.get("status") != "technical_complete_pending_visual_review" or \
            technical.get("status") != "technical_complete_pending_visual_and_protein_validation" or \
            technical.get("patient") != "CRC04" or \
            technical.get("channels") != list(CHANNELS) or \
            any(prep[key] != raw[key] for key in ("x", "y", "size_px", "mpp_x", "mpp_y")) or \
            (prep["size_px"], prep["mpp_x"], prep["mpp_y"]) != (2048, .325, .325):
        raise ValueError("Pilot technical completion or registered ORION source changed")
    if sha(input_case / "he.ome.tif") != technical["input_rgb_sha256"]:
        raise ValueError("Registered source RGB hash changed")
    with Image.open(input_case / "he.ome.tif") as handle:
        he = handle.convert("RGB")
    if he.size != (2048, 2048):
        raise ValueError("Unexpected registered H&E dimensions")
    if tuple(x["marker"] for x in raw["channels"]) != tuple(PAIRS[name][0] for name in PAIRS):
        raise ValueError("Measured channel order changed")
    return he, raw


def compare_native(measured: np.ndarray, virtual: np.ndarray, he: Image.Image) -> dict:
    """Compare untransformed scores on registered source coordinates.

    The virtual array is bilinearly resampled to the 0.325-µm source grid.
    Fixed stride four limits CPU work; each rank uses the same tissue pixels.
    """
    if measured.shape != (2048, 2048) or virtual.shape != (2048, 2048):
        raise ValueError("Registered marker shapes changed")
    rgb = np.asarray(he)[::4, ::4].astype(np.int16)
    tissue = (rgb.mean(axis=-1) < 240) & (np.ptp(rgb, axis=-1) > 12)
    x = measured[::4, ::4][tissue].astype(np.float32)
    y = virtual[::4, ::4][tissue].astype(np.float32)
    if len(x) < 10000 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Insufficient variable tissue pixels")
    rho = float(spearmanr(x, y).statistic)
    if not np.isfinite(rho):
        raise ValueError("Nonfinite raw-pixel rank association")
    return {"sampled_tissue_pixels": len(x),
            "pixel_spearman_rho": rho,
            "sampling_stride_source_pixels": 4,
            "scope": "One CRC04 window; raw spatial pixels only, no cell tables"}


def thumb(image: Image.Image, side: int = 380) -> Image.Image:
    item = image.copy()
    item.thumbnail((side, side), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(item, ((side-item.width)//2, (side-item.height)//2))
    return canvas


def run(input_case: Path, measured_root: Path, pilot_root: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Raw marker review must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    he, measured_source = source_checks(input_case, measured_root, pilot_root)
    store = zarr.open(str(pilot_root / "CRC04" / "gigatime" / "gigatime_probs.zarr"), mode="r")
    data = store["0"]
    if data.shape[0] != len(CHANNELS) or data.dtype != np.uint16 or \
            tuple(store.attrs["gigatime"]["store_channels"]) != CHANNELS:
        raise ValueError("Pilot native marker store identity changed")
    output.mkdir(parents=True)
    he.save(output / "CRC04-HE.webp", "WEBP", quality=88)
    paired = []
    all_channels = []
    paired_thumbs = []
    virtual_thumbs = []
    source_by_name = {x["marker"]: x for x in measured_source["channels"]}
    for index, marker in enumerate(CHANNELS):
        native = np.asarray(data[index], dtype=np.uint16)
        virtual = zoom(native.astype(np.float32),
                       (2048/native.shape[0], 2048/native.shape[1]),
                       order=1)
        if virtual.shape != (2048, 2048):
            raise ValueError(f"Virtual registered grid changed: {marker}")
        vd = Image.fromarray(display(virtual, virtual=True), "L")
        vd.save(output / f"CRC04-virtual-{marker}.webp", "WEBP", quality=88)
        virtual_thumbs.append((marker, thumb(vd.convert("RGB"), 256)))
        all_channels.append({"marker": marker,
                             "native_min": int(native.min()), "native_max": int(native.max()),
                             "native_nonconstant": bool(native.min() < native.max())})
        if marker not in PAIRS:
            continue
        measured_name, mapping = PAIRS[marker]
        path = measured_root / "CRC04" / f"{measured_name}.tif"
        if sha(path) != source_by_name[measured_name]["sha256"]:
            raise ValueError(f"Registered measured source hash changed: {measured_name}")
        measured = tifffile.imread(path)
        if measured.shape != (2048, 2048) or measured.dtype != np.uint16:
            raise ValueError(f"Measured spatial channel changed: {measured_name}")
        md = Image.fromarray(display(measured, virtual=False), "L")
        md.save(output / f"CRC04-measured-{measured_name}.webp", "WEBP", quality=88)
        overlap = Image.fromarray(np.stack((np.asarray(md), np.asarray(vd),
                                            np.asarray(md)), axis=-1), "RGB")
        overlap.save(output / f"CRC04-overlap-{marker}.webp", "WEBP", quality=88)
        score = compare_native(measured, virtual, he)
        paired.append({"virtual": marker, "measured": measured_name,
                       "mapping": mapping, **score})
        paired_thumbs.append((f"{marker} / {measured_name}: ρ={score['pixel_spearman_rho']:+.3f}",
                              [thumb(he), thumb(md.convert("RGB")),
                               thumb(vd.convert("RGB")), thumb(overlap)]))
    sheet = Image.new("RGB", (4*380, len(paired_thumbs)*(380+32)), "white")
    pen = ImageDraw.Draw(sheet)
    for row, (title, images) in enumerate(paired_thumbs):
        y = row*(380+32)
        pen.text((8, y+8), title, fill="black")
        for col, image in enumerate(images):
            sheet.paste(image, (col*380, y+32))
    sheet.save(output / "CRC04-paired-contact.webp", "WEBP", quality=86)
    overview = Image.new("RGB", (6*256, 4*(256+28)), "white")
    pen = ImageDraw.Draw(overview)
    for index, (name, image) in enumerate(virtual_thumbs):
        row, col = divmod(index, 6)
        x, y = col*256, row*(256+28)
        overview.paste(image, (x, y+28))
        pen.text((x+8, y+8), name, fill="black")
    overview.save(output / "CRC04-all-23-virtual.webp", "WEBP", quality=86)
    report = {"status": "pending_manual_visual_review", "patient": "CRC04",
              "source_mpp_um": .325, "model_mpp_um": .25,
              "source_rgb_sha256": sha(input_case / "he.ome.tif"),
              "channels": all_channels, "paired_markers": paired,
              "display": "Measured asinh and virtual logit, each 1st–99.7th percentile; visualization only. Magenta measured, green virtual in overlap.",
              "limit": "One previously QC-passing ORION field. Pixel rho does not establish cell-level protein validity or cross-patient association. Each image and shadow pattern requires manual review before site publication."}
    (output / "REPORT.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-case", type=Path, required=True)
    ap.add_argument("--measured-root", type=Path, required=True)
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps({"patient": run(a.input_case, a.measured_root,
                                     a.pilot_root, a.output)["patient"]}))
