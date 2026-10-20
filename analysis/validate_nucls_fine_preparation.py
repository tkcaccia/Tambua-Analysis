#!/usr/bin/env python3
"""Independently validate prepared NuCLS RGB calibration and target polygons."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


CLASSES = {"tumor_epithelial", "lymphocyte", "macrophage", "plasma_cell"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mpp(ome: str) -> tuple[float, float, str, str]:
    root = ET.fromstring(ome)
    pixels = next((x for x in root.iter() if x.tag.endswith("Pixels")), None)
    if pixels is None:
        raise ValueError("Missing OME Pixels element")
    return (float(pixels.attrib["PhysicalSizeX"]),
            float(pixels.attrib["PhysicalSizeY"]),
            pixels.attrib["PhysicalSizeXUnit"],
            pixels.attrib["PhysicalSizeYUnit"])


def run(plan_path: Path, prepared: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    done = json.loads((prepared/"PREP_COMPLETE.json").read_text())
    if plan["status"] != "fixed_source_cohort_no_inference" or \
            done["status"] != "source_prepared_no_inference" or \
            done["plan_sha256"] != sha(plan_path) or len(done["records"]) != 192:
        raise ValueError("Frozen source plan or completion identity changed")
    planned = {r["fov_id"]:r for r in plan["selected"]}
    records = {r["fov_id"]:r for r in done["records"]}
    if set(planned) != set(records) or len(records) != 192:
        raise ValueError("Prepared FOV set differs from source plan")
    patients = {"train":set(), "test":set()}
    counts = {"train":Counter(), "test":Counter()}
    for fov, saved in records.items():
        item = planned[fov]
        folder = prepared/fov
        if saved["split"] != item["split"] or \
                saved["patient_id"] != item["patient_id"] or \
                saved["source_mpp_um"] != .2 or \
                saved["target_model_mpp_um"] != .25:
            raise ValueError(f"Patient/split/scale mismatch: {fov}")
        patients[item["split"]].add(item["patient_id"])
        rgb_path = folder/"he.ome.tif"
        if sha(rgb_path) != saved["prepared_rgb_sha256"]:
            raise ValueError(f"RGB hash changed: {fov}")
        with tifffile.TiffFile(rgb_path) as tif:
            if tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB or \
                    tif.pages[0].shape != (item["height"],item["width"],3) or \
                    tif.pages[0].dtype != np.uint8 or \
                    mpp(tif.ome_metadata) != (.2,.2,"µm","µm"):
                raise ValueError(f"RGB format or physical size changed: {fov}")
        with tifffile.TiffFile(folder/"expert_target_nuclei.ome.tif") as tif:
            if tif.pages[0].shape != (item["height"],item["width"]) or \
                    tif.pages[0].dtype != np.uint32 or \
                    mpp(tif.ome_metadata) != (.2,.2,"µm","µm"):
                raise ValueError(f"Target mask grid/scale changed: {fov}")
            labels = tif.asarray()
        with tifffile.TiffFile(folder/"excluded_polyline_regions.ome.tif") as tif:
            if tif.pages[0].shape != labels.shape or \
                    mpp(tif.ome_metadata) != (.2,.2,"µm","µm"):
                raise ValueError(f"Excluded regions grid/scale changed: {fov}")
            excluded = tif.asarray()
        if np.any((labels>0)&(excluded>0)):
            raise ValueError(f"Typed mask overlaps excluded annotations: {fov}")
        with (folder/"expert_types.csv").open(newline="") as stream:
            target = list(csv.DictReader(stream))
        ids = [int(r["label_id"]) for r in target]
        observed = set(int(v) for v in np.unique(labels) if v)
        if len(ids) != len(set(ids)) or set(ids) != observed or \
                len(ids) != saved["typed_instance_count"]:
            raise ValueError(f"Typed CSV/mask IDs disagree: {fov}")
        local = Counter()
        for row in target:
            kind = row["cell_type"]
            if kind not in CLASSES or \
                    int(row["segmented_area_native_px"]) != \
                    int(np.count_nonzero(labels==int(row["label_id"]))):
                raise ValueError(f"Typed label or area changed: {fov}")
            local[kind] += 1
        if dict(local) != saved["target_polyline_counts_after_geometry_qc"]:
            raise ValueError(f"Recorded typed counts disagree: {fov}")
        counts[item["split"]].update(local)
    if patients["train"] & patients["test"] or \
            len(patients["train"]) != 59 or len(patients["test"]) != 12 or \
            counts["train"]["macrophage"] < 350 or \
            counts["test"]["macrophage"] < 100:
        raise ValueError("NuCLS patient separation or macrophage coverage failed")
    report = {"status":"validated_source_preparation_no_inference",
              "cases":len(records),"source_mpp_um":.2,"model_target_mpp_um":.25,
              "split":{name:{"images":sum(r["split"]==name for r in records.values()),
                              "patients":len(patients[name]),
                              "typed_polylines":dict(counts[name])}
                       for name in ("train","test")},
              "checks":"Every prepared OME RGB/target/excluded mask has matching native dimensions and 0.2-µm pixel size; typed CSV IDs, areas and classes match nonoverlapping mask pixels; RGB hashes and patient split match the frozen source plan.",
              "limit":"Validated data preparation only. No model inference, independent nuclear QC, classifier prediction, or balanced-accuracy estimate."}
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(output.suffix+".tmp")
    tmp.write_text(json.dumps(report,indent=2)+"\n")
    tmp.replace(output)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    x=run(a.plan,a.prepared,a.output)
    print(json.dumps({"status":x["status"],"cases":x["cases"],"split":x["split"]}))
