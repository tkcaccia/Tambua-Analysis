#!/usr/bin/env python3
"""Extract selected NuCLS RGB fields and typed nuclear polylines at native MPP.

Data-only CPU preparation. No GigaTIME, StarDist, model training, or scoring.
Convenient NuCLS raster masks are deliberately ignored because they conflate
rectangle boxes with segmented polylines.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import tifffile

from audit_nucls_fine_cell_types import SHARD_SHA256, TARGET, sha


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2)+"\n")
    tmp.replace(path)


def polygon_array(width: int, height: int, xs: list[int], ys: list[int]) -> np.ndarray:
    if len(xs) != len(ys) or len(xs) < 3 or len(set(zip(xs, ys))) < 3:
        return np.zeros((height, width), dtype=bool)
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon(list(zip(xs, ys)), fill=1)
    return np.asarray(image, dtype=bool)


def prepare_one(item: dict, planned: dict, output: Path) -> dict:
    name = planned["fov_id"]
    if item["fov_id"] != name:
        raise ValueError(f"Parquet row/FOV mismatch: {name}")
    image_bytes = item["image"]["bytes"]
    if not image_bytes:
        raise ValueError(f"Missing encoded RGB: {name}")
    with Image.open(io.BytesIO(image_bytes)) as source:
        rgb = np.asarray(source.convert("RGB"))
    height, width = rgb.shape[:2]
    if rgb.dtype != np.uint8 or (height, width) != \
            (planned["height"], planned["width"]):
        raise ValueError(f"Decoded RGB shape/type mismatch: {name}")
    annotations = list(csv.DictReader(io.StringIO(item["annotations_csv"])))
    label_map = np.zeros((height, width), dtype=np.uint32)
    excluded = np.zeros((height, width), dtype=np.uint8)
    typed_rows = []
    reasons = Counter()
    for row in annotations:
        if row["type"] != "polyline":
            reasons["non_polyline_"+row["type"]] += 1
            continue
        xs = [int(value) for value in row["coords_x"].split(",")]
        ys = [int(value) for value in row["coords_y"].split(",")]
        mask = polygon_array(width, height, xs, ys)
        pixels = int(mask.sum())
        if pixels < 4:
            reasons["degenerate_or_outside_polyline"] += 1
            continue
        cell_type = TARGET.get(row["main_classification"])
        if cell_type is None:
            conflicting = set(int(v) for v in np.unique(label_map[mask]) if v)
            for label_id in conflicting:
                prior = label_map == label_id
                excluded[prior] = 1
                label_map[prior] = 0
            reasons["target_overlaps_excluded_polyline"] += len(conflicting)
            excluded[mask] = 1
            reasons["excluded_fine_class"] += 1
            continue
        if np.any(label_map[mask] > 0) or np.any(excluded[mask] > 0):
            conflicting = set(int(v) for v in np.unique(label_map[mask]) if v)
            for label_id in conflicting:
                prior = label_map == label_id
                excluded[prior] = 1
                label_map[prior] = 0
            reasons["overlapping_target_polyline"] += 1
            excluded[mask] = 1
            continue
        label_id = len(typed_rows)+1
        label_map[mask] = label_id
        typed_rows.append({"label_id": label_id,
                           "cell_type": cell_type,
                           "source_row_id": row[""],
                           "main_classification": row["main_classification"],
                           "raw_classification": row["raw_classification"],
                           "segmented_area_native_px": pixels})
    # Earlier conflicting objects are erased in full; only intact target
    # polygons remain available to cell matching and intensity quantification.
    retained = set(int(value) for value in np.unique(label_map) if value)
    typed_rows = [r for r in typed_rows if r["label_id"] in retained]
    target_counter = Counter(r["cell_type"] for r in typed_rows)
    for row in typed_rows:
        row["segmented_area_native_px"] = int(np.count_nonzero(
            label_map == row["label_id"]))
    folder = output/name
    folder.mkdir()
    meta = {"axes": "YXS", "PhysicalSizeX": .2, "PhysicalSizeY": .2,
            "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    tifffile.imwrite(folder/"he.ome.tif", rgb, photometric="rgb", ome=True,
                     metadata=meta)
    mask_meta = {"axes": "YX", "PhysicalSizeX": .2, "PhysicalSizeY": .2,
                 "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    tifffile.imwrite(folder/"expert_target_nuclei.ome.tif", label_map,
                     photometric="minisblack", ome=True, metadata=mask_meta)
    tifffile.imwrite(folder/"excluded_polyline_regions.ome.tif", excluded,
                     photometric="minisblack", ome=True, metadata=mask_meta)
    with (folder/"expert_types.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label_id", "cell_type",
            "source_row_id", "main_classification", "raw_classification",
            "segmented_area_native_px"))
        writer.writeheader()
        writer.writerows(typed_rows)
    with tifffile.TiffFile(folder/"he.ome.tif") as tif:
        ome = tif.ome_metadata or ""
        if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB or \
                'PhysicalSizeX="0.2"' not in ome or \
                'PhysicalSizeY="0.2"' not in ome:
            raise ValueError(f"OME RGB/physical-scale roundtrip failed: {name}")
    return {"fov_id": name, "patient_id": planned["patient_id"],
            "split": planned["split"], "height": height, "width": width,
            "source_mpp_um": .2, "target_model_mpp_um": .25,
            "encoded_rgb_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "prepared_rgb_sha256": sha(folder/"he.ome.tif"),
            "target_polyline_counts_before_geometry_qc":
            {k:v for k,v in planned["target_polyline_counts"].items()
             if not k.startswith("excluded_")},
            "target_polyline_counts_after_geometry_qc": dict(target_counter),
            "typed_instance_count": len(typed_rows),
            "excluded_reasons": dict(reasons)}


def run(source: Path, plan_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    if plan["status"] != "fixed_source_cohort_no_inference" or \
            len(plan["selected"]) != 192:
        raise ValueError("Expected frozen 192-image NuCLS plan")
    selected = {r["fov_id"]: r for r in plan["selected"]}
    if len(selected) != 192:
        raise ValueError("Duplicate selected FOV")
    output.mkdir(parents=True)
    records = []
    for name, expected in SHARD_SHA256.items():
        path = source/name
        if sha(path) != expected:
            raise ValueError(f"NuCLS source shard changed: {name}")
        rows = {r["source_row"]: r for r in selected.values()
                if r["source_shard"] == name}
        if not rows:
            continue
        table = pq.read_table(path,
                              columns=["fov_id", "image", "annotations_csv"])
        for index, planned in rows.items():
            records.append(prepare_one(table.slice(index,1).to_pylist()[0],
                                       planned, output))
    if len(records) != 192:
        raise ValueError("Prepared NuCLS cohort count changed")
    report = {"status": "source_prepared_no_inference", "cases": len(records),
              "plan_sha256": sha(plan_path), "source_shard_sha256": SHARD_SHA256,
              "physical_resolution": "0.2 µm/pixel in source OME-TIFF; future GigaTIME must resample to 0.25 µm/pixel, with no claim of gained detail",
              "annotation_geometry": "Only intact target-class polylines become typed instances; rectangles, points, ambiguous and all conflicting polygons are excluded from the typed target mask",
              "records": sorted(records,key=lambda r:r["fov_id"]),
              "limit": "Data-only preparation. No GigaTIME/StarDist output, technical QC, classifier, score or public cell-type result."}
    atomic_json(output/"PREP_COMPLETE.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    x = run(a.source,a.plan,a.output)
    print(json.dumps({"cases":x["cases"],"status":x["status"]}))
