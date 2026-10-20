#!/usr/bin/env python3
"""CPU-only paired HEMIT TIFF audit; never runs virtual staining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import tifffile


PATCH = re.compile(r"_patch_(\d+)_(\d+)\.tif$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(tag: object) -> float | None:
    if tag is None:
        return None
    value = getattr(tag, "value", tag)
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1]) if value[1] else None
    return float(value)


def inspect(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = page.tags
        unit_tag = tags.get("ResolutionUnit")
        unit = (getattr(unit_tag.value, "name", str(unit_tag.value))
                if unit_tag else None)
        xres = ratio(tags.get("XResolution"))
        yres = ratio(tags.get("YResolution"))
        micron_unit = 25400 if unit and "INCH" in unit.upper() else 10000 if unit and "CENTIMETER" in unit.upper() else None
        # TIFF resolution tags are often merely display DPI; record, do not
        # silently treat them as trusted microscope calibration.
        inferred = [micron_unit / value if micron_unit and value else None for value in (xres, yres)]
        data = page.asarray()
        if data.ndim != 3 or data.shape[:2] != (1024, 1024) or data.shape[2] != 3:
            raise ValueError(f"Unexpected TIFF shape: {path} {data.shape}")
        center = data[256:768, 256:768]
        return {
            "shape": list(data.shape), "dtype": str(data.dtype),
            "photometric": str(page.photometric),
            "samples_per_pixel": int(page.samplesperpixel),
            "resolution_unit": unit, "x_resolution": xres,
            "y_resolution": yres, "tag_implied_um_per_pixel_unverified": inferred,
            "channel_min": [float(x) for x in center.min(axis=(0, 1))],
            "channel_median": [float(x) for x in np.median(center, axis=(0, 1))],
            "channel_max": [float(x) for x in center.max(axis=(0, 1))],
        }


def run(root: Path, out: Path, sample_count: int, ids_path: Path) -> dict:
    images = root / "test" / "input"
    labels = root / "test" / "label"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("Expected HEMIT/test/{input,label} in extracted archive")
    input_names = {p.name for p in images.glob("*.tif")}
    label_names = {p.name for p in labels.glob("*.tif")}
    if input_names != label_names:
        raise ValueError(f"Unpaired TIFF files: {len(input_names-label_names)} inputs and {len(label_names-input_names)} labels")
    if len(input_names) != 945:
        raise ValueError(f"Expected 945 official HEMIT test pairs, found {len(input_names)}")
    nonoverlap = []
    for name in sorted(input_names):
        match = PATCH.search(name)
        if not match:
            raise ValueError(f"Unrecognized HEMIT patch name: {name}")
        if int(match.group(1)) % 2 == 0 and int(match.group(2)) % 2 == 0:
            nonoverlap.append(name)
    published_ids = [name.strip() for name in ids_path.read_text().splitlines() if name.strip()]
    if len(published_ids) != 292 or len(set(published_ids)) != 292 or \
       not set(published_ids).issubset(set(nonoverlap)):
        raise ValueError("Published 292-pair nonoverlap selection does not match source archive")
    chosen = published_ids[:sample_count]
    records = []
    for name in chosen:
        image = images / name
        label = labels / name
        records.append({
            "id": name, "input_sha256": sha256(image),
            "label_sha256": sha256(label), "input": inspect(image),
            "label": inspect(label),
        })
    result = {
        "status": "cpu_input_audit_only_no_gigatime_inference",
        "source": "https://data.mendeley.com/datasets/3gx53zm49d/1",
        "official_test_pairs": len(input_names),
        "even_even_nonoverlapping_candidates": len(nonoverlap),
        "published_nonoverlap_nonempty_pairs": len(published_ids),
        "published_ids_sha256": sha256(ids_path),
        "inspected_pairs": records,
        "physical_scale_status": "unverified_until_source_and_tiff_metadata_are_reconciled",
        "provisional_label_band_order": ["Pan-CK", "CD3", "DAPI"],
        "channel_semantics_status": "MIPHEI_preprocessing_mapping_requires_pixel_review",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--ids", type=Path,
                        default=Path(__file__).with_name("hemit_test_no_overlap_no_empty_ids.txt"))
    args = parser.parse_args()
    if not 1 <= args.sample_count <= 292:
        parser.error("--sample-count must be between 1 and 292")
    result = run(args.root, args.output, args.sample_count, args.ids)
    print(json.dumps({key: value for key, value in result.items() if key != "inspected_pairs"}, indent=2))


if __name__ == "__main__":
    main()
