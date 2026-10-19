"""Spatial plausibility check for GigaTIME DAPI against H&E nuclear morphology.

The H&E hematoxylin channel is a proxy, not a reference nuclear annotation.
Always inspect the saved overlay before accepting a case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from skimage.color import rgb2hed
import tifffile


def ome_mpp(path: str | Path) -> tuple[float, tuple[int, int], np.ndarray]:
    with tifffile.TiffFile(path) as source:
        if not source.ome_metadata:
            raise ValueError("H&E image requires OME physical pixel metadata")
        root = ET.fromstring(source.ome_metadata)
        pixels = next(element for element in root.iter() if element.tag.endswith("Pixels"))
        if pixels.get("PhysicalSizeXUnit", "µm") not in ("µm", "um") or \
           pixels.get("PhysicalSizeYUnit", "µm") not in ("µm", "um"):
            raise ValueError("Pixel spacing must be in micrometres")
        x, y = float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])
        if abs(x - y) > 1e-6 or x <= 0:
            raise ValueError("H&E pixel spacing must be positive and isotropic")
        rgb = np.asarray(source.series[0].asarray())
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected RGB H&E image")
    return x, tuple(rgb.shape[:2]), rgb


def inspect(he_path: str | Path, dapi_path: str | Path, cell_mask_path: str | Path,
            manifest_path: str | Path, output_dir: str | Path) -> dict:
    source_mpp, source_shape, he = ome_mpp(he_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    target_mpp = float(metadata["effective_mpp"])
    declared_source_mpp = float(metadata["source_mpp"])
    if abs(target_mpp - .25) > 1e-6 or abs(declared_source_mpp - source_mpp) > 1e-6:
        raise ValueError("GigaTIME resolution contract does not match the H&E source and 0.25 µm/pixel target")
    if tuple(metadata["original_shape_yx"]) != source_shape:
        raise ValueError("GigaTIME source shape differs from H&E")
    with Image.open(dapi_path) as image:
        dapi = np.asarray(image.convert("L"), dtype=np.uint8)
    target_shape = tuple(int(v) for v in metadata["inference_shape_yx"])
    if dapi.shape != target_shape or tuple(manifest["image_shape_yx"]) != target_shape:
        raise ValueError("DAPI image dimensions differ from the calibrated inference grid")
    cells = np.asarray(tifffile.imread(cell_mask_path))
    if cells.shape != source_shape:
        raise ValueError("Whole-cell mask shape differs from H&E")
    scaled_he = np.asarray(Image.fromarray(he).resize((target_shape[1], target_shape[0]),
                                                     Image.Resampling.BILINEAR))
    scaled_cells = np.asarray(Image.fromarray(cells).resize((target_shape[1], target_shape[0]),
                                                           Image.Resampling.NEAREST)) > 0
    hematoxylin = rgb2hed(scaled_he)[..., 0]
    if int(scaled_cells.sum()) < 10000:
        raise ValueError("Too few cell pixels for DAPI overlap QC")
    h_cut = float(np.quantile(hematoxylin[scaled_cells], .75))
    d_cut = float(np.quantile(dapi[scaled_cells], .75))
    # Strict inequality avoids declaring every background pixel "high" when
    # a sparse positive channel has a zero-valued 75th percentile.
    h_high = (hematoxylin > h_cut) & scaled_cells
    d_high = (dapi > d_cut) & scaled_cells
    observed = int(np.count_nonzero(h_high & d_high))
    area = int(scaled_cells.sum())
    expected = int(h_high.sum()) * int(d_high.sum()) / area
    report = dict(source_mpp_um=source_mpp, model_mpp_um=target_mpp,
                  source_shape_yx=list(source_shape), inference_shape_yx=list(target_shape),
                  whole_cell_pixels=area,
                  hematoxylin_high_fraction=float(h_high.sum() / area),
                  dapi_high_fraction=float(d_high.sum() / area),
                  observed_high_overlap_pixels=observed,
                  random_independence_expected_overlap_pixels=float(expected),
                  overlap_enrichment=(float(observed / expected) if expected else None),
                  interpretation="H&E hematoxylin is a nuclear proxy; this is not expert-mask Dice or marker validation.",
                  review_status="manual visual review required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    glow = np.asarray(dapi, dtype=np.float32) / 255.0
    overlay = scaled_he.astype(np.float32)
    overlay[..., 0] = np.maximum(overlay[..., 0], glow * 255)
    overlay[..., 2] = np.maximum(overlay[..., 2], glow * 255)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output / "he_dapi_overlay.png")
    Image.fromarray((d_high.astype(np.uint8) * 255)).save(output / "dapi_high.png")
    (output / "qc_dapi.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he", required=True)
    parser.add_argument("--dapi", required=True)
    parser.add_argument("--cell-mask", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.he, args.dapi, args.cell_mask, args.manifest,
                             args.output_dir), indent=2))


if __name__ == "__main__":
    main()
