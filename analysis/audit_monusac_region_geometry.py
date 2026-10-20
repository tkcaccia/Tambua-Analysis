"""Describe MoNuSAC expert-polygon areas without changing the reference masks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import tifffile


def audit(prepared: Path, expected_images: int = 120) -> dict:
    areas: dict[str, list[int]] = defaultdict(list)
    unusual = []
    images = 0
    for folder in sorted(prepared.iterdir()):
        if not folder.is_dir() or not (folder / "expert_nuclei.ome.tif").is_file():
            continue
        images += 1
        provenance = json.loads((folder / "provenance.json").read_text())
        mask = tifffile.imread(folder / "expert_nuclei.ome.tif")
        counts = np.bincount(mask.ravel())
        with (folder / "expert_types.csv").open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                label_id = int(row["label_id"])
                area = int(counts[label_id]) if label_id < len(counts) else 0
                cell_type = row["cell_type"]
                areas[cell_type].append(area)
                if cell_type == "macrophage" and area > 5000:
                    unusual.append(dict(sample_id=folder.name, label_id=label_id,
                                        source_class=cell_type, pixels=area,
                                        area_um2=round(area * provenance["mpp"] ** 2, 1)))
    if images != expected_images:
        raise ValueError(f"Expected {expected_images} prepared images, got {images}")
    summary = {}
    for cell_type, values in sorted(areas.items()):
        arr = np.asarray(values)
        summary[cell_type] = dict(count=len(values), median_px=float(np.median(arr)),
                                  p99_px=float(np.percentile(arr, 99)),
                                  max_px=int(arr.max()), greater_than_5000_px=int((arr > 5000).sum()))
    return dict(scope="Source expert-region geometry; no exclusions or corrected labels",
                images=images, regions=sum(len(v) for v in areas.values()),
                threshold_note="Macrophage >5000 px is a manual-review flag, not an exclusion rule",
                by_class=summary, large_source_macrophage_regions=unusual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=120)
    args = parser.parse_args()
    result = audit(args.prepared, args.expected_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"images": result["images"], "regions": result["regions"],
                      "large_source_macrophage_regions": len(result["large_source_macrophage_regions"])}))
