#!/usr/bin/env python3
"""CPU-only, lossless RGB stain-reference preflight; never runs GigaTIME.

The profile uses unlabelled author-supplied H&E test patches. Output PNGs are
visual diagnostics, not validated inference inputs. They carry no microscopy
calibration; the source OME-TIFF and its pixel spacing remain authoritative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import lab2rgb, rgb2lab
import tifffile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.mode != "RGB":
            raise ValueError(f"Input must declare RGB, not {opened.mode}: {path}")
        image = np.asarray(opened, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 RGB: {path}")
    return image


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    lo = rgb.min(axis=2).astype(np.int16)
    chroma = rgb.max(axis=2).astype(np.int16) - lo
    return (lo < 225) & (chroma > 15)


def lab_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    tissue = tissue_mask(rgb)
    n = int(np.count_nonzero(tissue))
    if n < 1000:
        raise ValueError(f"Too little stained tissue for whole-image normalization: {n} pixels")
    lab = rgb2lab(rgb)
    selected = lab[tissue]
    return selected.mean(axis=0), selected.std(axis=0), n


def reference_profile(reference_paths: list[Path]) -> dict:
    if len(reference_paths) < 10:
        raise ValueError("At least ten independent H&E reference patches are required")
    entries = []
    means, stds = [], []
    for path in sorted(reference_paths):
        rgb = read_rgb(path)
        mean, std, n = lab_stats(rgb)
        means.append(mean)
        stds.append(std)
        entries.append({"name": path.name, "sha256": file_sha256(path),
                        "shape_yx": list(rgb.shape[:2]), "tissue_pixels": n})
    target_mean = np.median(np.stack(means), axis=0)
    target_std = np.median(np.stack(stds), axis=0)
    if np.any(target_std < 1):
        raise ValueError("Reference stain variation is too small")
    return {"method": "bounded_luminance_only_tissue_stats_v2",
            "reference_images": len(entries), "reference": entries,
            "target_lab_mean": target_mean.tolist(),
            "target_lab_std": target_std.tolist(),
            "notes": "Median of per-patch tissue LAB means and standard deviations. Only the L target is used for the bounded preview transform; no cell labels or model outputs are used."}


def normalize(rgb: np.ndarray, profile: dict) -> tuple[np.ndarray, dict]:
    mean, std, n = lab_stats(rgb)
    target_mean = np.asarray(profile["target_lab_mean"], dtype=np.float64)
    lab = rgb2lab(rgb)
    changed = lab.copy()
    lightness_shift = float(np.clip(target_mean[0] - mean[0], -25.0, 25.0))
    changed[..., 0] = np.clip(changed[..., 0] + lightness_shift, 0, 100)
    recolored = lab2rgb(changed)
    # Pure-white background stays white. The continuous optical-density weight
    # avoids introducing a hard ring at pale tissue/background transitions.
    opacity = np.clip((255.0 - rgb.min(axis=2).astype(np.float32)) / 20.0, 0, 1)
    blended = recolored * opacity[..., None] + (rgb / 255.0) * (1 - opacity[..., None])
    output = np.rint(np.clip(blended, 0, 1) * 255).astype(np.uint8)
    post_mean, post_std, _ = lab_stats(output)
    report = {"tissue_pixels": n, "source_lab_mean": mean.tolist(),
              "source_lab_std": std.tolist(), "bounded_lightness_shift": lightness_shift,
              "source_to_reference_chroma_distance": float(np.linalg.norm(mean[1:] - target_mean[1:])),
              "chroma_out_of_reference_range": bool(np.linalg.norm(mean[1:] - target_mean[1:]) > 20),
              "normalized_lab_mean": post_mean.tolist(),
              "normalized_lab_std": post_std.tolist(),
              "pure_white_unchanged": bool(np.all(output[np.all(rgb == 255, axis=2)] == 255))}
    return output, report


def thumbnail(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, "white")
    tile.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return tile


def write_calibrated_ome(source: Path, rgb: np.ndarray, destination: Path) -> dict:
    with tifffile.TiffFile(source) as tif:
        xml = tif.ome_metadata
        if xml is None or tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB:
            raise ValueError(f"Calibrated RGB OME-TIFF required: {source}")
        pixels = ET.fromstring(xml).find(".//{*}Pixels")
        if pixels is None:
            raise ValueError(f"OME pixel metadata missing: {source}")
        mx = float(pixels.attrib["PhysicalSizeX"])
        my = float(pixels.attrib["PhysicalSizeY"])
        ux = pixels.attrib.get("PhysicalSizeXUnit")
        uy = pixels.attrib.get("PhysicalSizeYUnit")
        if ux not in {"µm", "um"} or uy not in {"µm", "um"} or \
                not (0 < mx <= 1 and 0 < my <= 1) or abs(mx / my - 1) > .01:
            raise ValueError(f"Untrusted microscopy spacing: {source}")
    tifffile.imwrite(destination, rgb, photometric="rgb", ome=True,
                     compression="deflate", metadata={"axes": "YXS",
                     "PhysicalSizeX": mx, "PhysicalSizeXUnit": "µm",
                     "PhysicalSizeY": my, "PhysicalSizeYUnit": "µm"})
    with tifffile.TiffFile(destination) as tif:
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB or \
                not np.array_equal(tif.asarray(), rgb):
            raise AssertionError("Lossless RGB OME-TIFF round trip failed")
        saved = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        if saved is None or abs(float(saved.attrib["PhysicalSizeX"]) - mx) > 1e-9 or \
                abs(float(saved.attrib["PhysicalSizeY"]) - my) > 1e-9:
            raise AssertionError("Microscopy spacing changed during OME export")
    return {"x_um_per_px": mx, "y_um_per_px": my, "compression": "lossless_deflate"}


def run(references: Path, inputs: list[Path], output: Path,
        *, write_ome: bool = False) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Stain preflight must explicitly hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    profile = reference_profile(sorted(references.glob("*_he.png")))
    output.mkdir(parents=True)
    rows = []
    contact = Image.new("RGB", (1000, 550 * len(inputs) + 50), "white")
    pen = ImageDraw.Draw(contact)
    pen.text((10, 10), "Scanner RGB", fill="black")
    pen.text((510, 10), "Reference-normalized preview", fill="black")
    for idx, path in enumerate(inputs):
        original = read_rgb(path)
        changed, metrics = normalize(original, profile)
        if changed.shape != original.shape:
            raise AssertionError("Normalization changed pixel coordinates")
        name = f"sample-{idx + 1:02d}"
        Image.fromarray(changed, mode="RGB").save(output / f"{name}-normalized.png")
        calibrated = None
        if write_ome:
            calibrated = write_calibrated_ome(path, changed,
                                              output / f"{name}-normalized.ome.tif")
        y = 50 + idx * 550
        pen.text((10, y), f"{name}  {path.name}", fill="black")
        contact.paste(thumbnail(original, (490, 500)), (0, y + 30))
        contact.paste(thumbnail(changed, (490, 500)), (500, y + 30))
        rows.append({"name": name, "source": str(path), "source_sha256": file_sha256(path),
                     "shape_yx": list(original.shape[:2]), "output": f"{name}-normalized.png",
                     "calibrated_ome": calibrated, **metrics})
    contact.save(output / "contact.webp", "WEBP", quality=90)
    report = {"status": "preflight_only", "device": "CPU", "profile": profile,
              "cases": rows, "model_inference": "not_run",
              "interpretation": "A visually plausible color transform is not evidence that GigaTIME DAPI or proteins improve. Source RGB, spatial dimensions and model input calibration must be reverified before any inference."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-calibrated-ome", action="store_true")
    args = parser.parse_args()
    result = run(args.references, args.input, args.output,
                 write_ome=args.write_calibrated_ome)
    print(json.dumps({"status": result["status"], "reference_images":
                      result["profile"]["reference_images"], "cases": len(result["cases"])}))
