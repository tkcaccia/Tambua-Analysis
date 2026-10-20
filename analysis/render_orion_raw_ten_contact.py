#!/usr/bin/env python3
"""CPU-only diagnostic contact sheets for ten registered ORION fields."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
import tifffile
import zarr

from render_orion_pixel_dapi import display
from score_orion_raw_nine import PAIRS, PATIENTS
from validate_orion_nine_raw_case import CHANNELS


def thumb(array: np.ndarray | Image.Image, side: int) -> Image.Image:
    item = array if isinstance(array, Image.Image) else Image.fromarray(array)
    return item.convert("RGB").resize((side, side), Image.Resampling.LANCZOS)


def render_case(patient: str, source: Path, measured_root: Path,
                predicted_root: Path, report: dict, output: Path) -> None:
    row_by_marker = {r["virtual"]: r for r in report["per_patient"] if r["patient"] == patient}
    if set(row_by_marker) != set(PAIRS):
        raise ValueError(f"Incomplete fixed raw-pixel scores: {patient}")
    with Image.open(source / patient / "he.ome.tif") as handle:
        he = handle.convert("RGB")
    z = zarr.open(str(predicted_root / patient / "gigatime" / "gigatime_probs.zarr"), mode="r")
    data = z["0"]
    if he.size != (2048, 2048) or data.shape[0] != 23 or \
       tuple(z.attrs["gigatime"]["store_channels"]) != CHANNELS:
        raise ValueError(f"Source or GigaTIME channel geometry changed: {patient}")
    he_thumb = thumb(he, 252)
    paired = Image.new("RGB", (4*252, len(PAIRS)*(252+30)), "white")
    pen = ImageDraw.Draw(paired)
    overview = Image.new("RGB", (6*224, 4*(224+26)), "white")
    over_pen = ImageDraw.Draw(overview)
    details = Image.new("RGB", (4*512, 3*(512+32)), "white")
    detail_pen = ImageDraw.Draw(details)
    names = list(PAIRS)
    for index, name in enumerate(CHANNELS):
        native = np.asarray(data[index], dtype=np.uint16)
        aligned = zoom(native.astype(np.float32),
                       (2048/native.shape[0], 2048/native.shape[1]), order=1)
        if aligned.shape != (2048, 2048):
            raise ValueError("Physical pixel alignment changed")
        virtual = display(aligned, virtual=True)
        vthumb = thumb(virtual, 224)
        r, c = divmod(index, 6)
        overview.paste(vthumb, (c*224, r*(224+26)+26))
        over_pen.text((c*224+6, r*(224+26)+5), name, fill="black")
        if name not in PAIRS:
            continue
        measured_name, _ = PAIRS[name]
        observed_raw = tifffile.imread(measured_root / patient / f"{measured_name}.tif")
        if observed_raw.shape != (2048, 2048):
            raise ValueError("Measured spatial size changed")
        measured = display(observed_raw, virtual=False)
        overlap = np.stack((measured, virtual, measured), axis=-1)
        rr = names.index(name)
        y = rr*(252+30)
        for col, item in enumerate((he_thumb, thumb(measured, 252),
                                    thumb(virtual, 252), thumb(overlap, 252))):
            paired.paste(item, (col*252, y+30))
        rho = row_by_marker[name]["pixel_spearman_rho"]
        pen.text((6, y+6), f"{name} / {measured_name}   raw tissue pixel rho={rho:+.3f}", fill="black")
        if name in ("DAPI", "CD68", "CK"):
            dr = ("DAPI", "CD68", "CK").index(name)
            crop = (768, 768, 1280, 1280)
            for col, item in enumerate((he.crop(crop), Image.fromarray(measured).crop(crop),
                                        Image.fromarray(virtual).crop(crop),
                                        Image.fromarray(overlap).crop(crop))):
                details.paste(item.convert("RGB"), (col*512, dr*(512+32)+32))
            detail_pen.text((8, dr*(512+32)+7), f"{name} / {measured_name}  center 512-pixel crop", fill="black")
    paired.save(output / f"{patient}-paired.webp", "WEBP", quality=87)
    overview.save(output / f"{patient}-all23.webp", "WEBP", quality=87)
    details.save(output / f"{patient}-detail.webp", "WEBP", quality=88)


def run(source: Path, measured: Path, pilot: Path, nine: Path,
        scores: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Contact sheet rendering must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(scores.read_text())
    if report["patients"] != list(PATIENTS) or len(report["per_patient"]) != 100:
        raise ValueError("Ten-patient raw pixel score is incomplete")
    output.mkdir(parents=True)
    for patient in PATIENTS:
        root = pilot if patient == "CRC04" else nine
        render_case(patient, source, measured, root, report, output)
    manifest = {"status": "diagnostic_contacts_pending_manual_review",
                "patients": list(PATIENTS), "source_mpp_um": .325, "model_mpp_um": .25,
                "channels_per_patient": 23, "paired_channels_per_patient": 10,
                "display": "Per-image measured asinh, virtual logit; 1st–99.7th percentile for visualization only. In overlaps, magenta is measured and green virtual.",
                "scoring": "Raw 16-bit values on a common source grid, tissue pixels sampled at stride four; no display transformation used for Spearman.",
                "limitation": "No protein-specific biological validity inferred from technical QC or display; manual visual review required."}
    (output / "CONTACTS.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--measured", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--nine", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.source, a.measured, a.pilot, a.nine, a.scores, a.output)))
