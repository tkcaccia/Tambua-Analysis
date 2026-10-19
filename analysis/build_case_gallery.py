"""Render aligned ORION case panels for the static results page.

The ORION instance mask is a published computational segmentation, not an
expert-drawn cell-type annotation. Marker JPEGs are displayed on the exact
physical field of view of the registered H&E crop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
import tifffile


FIELDS = {
    "glands": (0, 0, 800, 800),
    "mixed": (1100, 200, 1900, 1000),
    "stroma": (750, 1100, 1550, 1900),
}
MARKERS = ("DAPI", "CK", "CD3", "CD68")


def build(he_path: Path, mask_path: Path, output_path: Path, manifest_path: Path,
          destination: Path) -> dict:
    he = tifffile.imread(he_path)
    mask = tifffile.imread(mask_path)
    meta = json.loads(manifest_path.read_text())["metadata"]
    if he.shape != (2048, 2048, 3) or mask.shape != he.shape[:2]:
        raise ValueError("Unexpected registered ORION image/mask geometry")
    if (float(meta["source_mpp"]), float(meta["effective_mpp"])) != (0.325, 0.25):
        raise ValueError("This gallery is only for the calibrated ORION case")
    scale = float(meta["source_mpp"]) / float(meta["effective_mpp"])
    destination.mkdir(parents=True, exist_ok=True)
    marker_images = {name: Image.open(output_path / "jpg_image" / f"{name}.jpg").convert("RGB")
                     for name in MARKERS}
    result = {"source": "ORION CRC33_01", "source_mpp_um": 0.325,
              "model_mpp_um": 0.25, "reference_kind": "published computational cell-instance mask",
              "fields": {}}
    for field, (left, top, right, bottom) in FIELDS.items():
        rgb = np.asarray(he[top:bottom, left:right]).copy()
        region_mask = mask[top:bottom, left:right]
        contour = find_boundaries(region_mask, mode="outer") & (region_mask > 0)
        overlay = rgb.copy()
        overlay[contour] = (0, 220, 200)
        outputs = {"he": Image.fromarray(rgb), "reference": Image.fromarray(overlay)}
        marker_box = tuple(round(value * scale) for value in (left, top, right, bottom))
        outputs.update({name.lower(): image.crop(marker_box) for name, image in marker_images.items()})
        dapi_gray = np.asarray(outputs["dapi"].convert("L").resize((800, 800), Image.Resampling.BILINEAR))
        strong = (dapi_gray > np.quantile(dapi_gray[region_mask > 0], .83)) & (region_mask > 0)
        tinted = rgb.astype(np.float32) * .95
        tinted[strong, 0] = np.maximum(tinted[strong, 0], 235)
        tinted[strong, 2] = np.maximum(tinted[strong, 2], 225)
        outputs["dapi_overlay"] = Image.fromarray(np.clip(tinted, 0, 255).astype(np.uint8))
        paths = {}
        for name, image in outputs.items():
            image = image.resize((640, 640), Image.Resampling.LANCZOS)
            filename = f"{field}-{name}.webp"
            image.save(destination / filename, "WEBP", quality=82, method=6)
            paths[name] = filename
        result["fields"][field] = {"source_box_xyxy": [left, top, right, bottom],
                                   "physical_width_um": (right - left) * 0.325,
                                   "images": paths}
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--he", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--gigatime-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.he, args.mask, args.gigatime_output, args.manifest,
                           args.destination), indent=2))


if __name__ == "__main__":
    main()
