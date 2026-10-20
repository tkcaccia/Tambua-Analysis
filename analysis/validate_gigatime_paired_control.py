#!/usr/bin/env python3
"""One bounded CPU diagnostic: official paired DAPI and one MoNuSAC stain pair.

This is an isolated research control, never a classifier score or public result.
The caller must have a separately authorized Chiamaka execution scope. It does
not submit any scheduler or GPU work and stops at the first error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
import tifffile
import torch


REVISION = "a6556030bdb8071857559e12b921620aa1bbac5b"
CHECKPOINT_SHA256 = "d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c"
OFFICIAL_HE_SHA256 = "ae5cc2df9710e69362a2b6c5f20a4e2388cbbcf556a9e2acb78267c843f2f61d"
OFFICIAL_LABELS_SHA256 = "ff857a3145d565497a2ca2992883ff8e6ef9279ab52b91954dac5589616bf347"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_code(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mpp_of(path: Path) -> float:
    with tifffile.TiffFile(path) as tif:
        if tif.ome_metadata is None or tif.pages[0].photometric != tifffile.PHOTOMETRIC.RGB:
            raise ValueError(f"Calibrated RGB OME-TIFF required: {path}")
        pixels = ET.fromstring(tif.ome_metadata).find(".//{*}Pixels")
        if pixels is None:
            raise ValueError(f"OME pixel metadata missing: {path}")
        x, y = float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])
        if pixels.attrib.get("PhysicalSizeXUnit") not in {"µm", "um"} or \
                pixels.attrib.get("PhysicalSizeYUnit") not in {"µm", "um"} or \
                not (0 < x <= 1 and 0 < y <= 1) or abs(x / y - 1) > .01:
            raise ValueError(f"Untrusted microscopy resolution: {path}")
        return (x + y) / 2


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"RGB source required: {path}")
        array = np.asarray(image, dtype=np.uint8)
    return array


def inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU control must hide all GPUs")
    if digest(args.official_he) != OFFICIAL_HE_SHA256 or \
            digest(args.official_labels) != OFFICIAL_LABELS_SHA256:
        raise ValueError("Author reference files differ from the integrity-checked archive")
    source = rgb(args.monusac_original)
    changed = rgb(args.monusac_normalized)
    with tifffile.TiffFile(args.stardist) as tif:
        labels = tif.asarray()
    he = rgb(args.official_he)
    if source.shape != changed.shape or source.shape[:2] != labels.shape or \
            source.size > 4_000_000 or he.shape != (556, 556, 3):
        raise ValueError("Diagnostic image geometry or size changed")
    old_mpp, new_mpp = mpp_of(args.monusac_original), mpp_of(args.monusac_normalized)
    if abs(old_mpp - new_mpp) > 1e-9 or abs(old_mpp / .25 - 1) > .05:
        raise ValueError("Normalized and source microscopy resolution differ")
    # This pickle is only read from the author's integrity-checked reference
    # archive, never an arbitrary user-supplied file.
    with gzip.open(args.official_labels, "rb") as stream:
        expert = pickle.load(stream)
    dapi_labels = np.asarray(expert["labels_dapi"])
    if dapi_labels.shape != (556, 556) or not np.any(dapi_labels > 0):
        raise ValueError("Author paired DAPI labels are missing")
    return ({"official_he": he, "official_labels": dapi_labels,
             "monusac_original": source, "monusac_normalized": changed,
             "monusac_stardist": labels, "source_mpp": old_mpp},
            {"official_he_sha256": digest(args.official_he),
             "official_labels_sha256": digest(args.official_labels),
             "monusac_original_sha256": digest(args.monusac_original),
             "monusac_normalized_sha256": digest(args.monusac_normalized),
             "stardist_sha256": digest(args.stardist), "source_mpp": old_mpp})


def score(dapi: np.ndarray, labels: np.ndarray, qc_module,
          reference_type: str) -> dict:
    scorer = qc_module.NuclearAlignmentScore(dapi.shape)
    scorer.add(dapi, labels, 0, 0)
    report = scorer.report()
    report["reference_type"] = reference_type
    if reference_type == "author_paired_measured_DAPI_nuclei":
        report["interpretation_limit"] = (
            "Reference nuclei come from the author's paired mIF sample after "
            "nearest-neighbor resizing; this is one positive-control patch, "
            "not validation of other channels or tissues."
        )
    return report


def preview(he: np.ndarray, dapi: np.ndarray, labels: np.ndarray, path: Path) -> None:
    low, high = np.quantile(dapi, [.01, .997])
    gray = np.rint(np.clip((dapi - low) / max(float(high - low), 1e-6), 0, 1) * 255).astype(np.uint8)
    overlay = np.repeat(gray[..., None], 3, axis=2)
    overlay[find_boundaries(labels > 0, mode="outer")] = (255, 60, 60)
    contact = Image.new("RGB", (he.shape[1] * 3, he.shape[0]))
    for idx, image in enumerate((he, np.repeat(gray[..., None], 3, axis=2), overlay)):
        contact.paste(Image.fromarray(image, mode="RGB"), (idx * he.shape[1], 0))
    contact.save(path, "PNG")


def official_inference(he: np.ndarray, model, runner) -> tuple[np.ndarray, np.ndarray]:
    # Exactly the author's notebook geometry: 556-pixel H&E resized to 512,
    # followed by four non-overlapping 256-pixel CNN windows.
    resized = cv2.resize(he, (512, 512), interpolation=cv2.INTER_LINEAR)
    output = np.empty((512, 512), dtype=np.float32)
    with torch.no_grad():
        for y in (0, 256):
            for x in (0, 256):
                patch = resized[y:y + 256, x:x + 256]
                tensor = runner.normalize_patch_batch(patch[None])
                output[y:y + 256, x:x + 256] = torch.sigmoid(model(tensor))[0, 0].numpy()
    return resized, output


def monusac_inference(he: np.ndarray, model, runner) -> np.ndarray:
    padded, positions = runner.pad_to_tiling_shape(he, 256, 128)
    accum, counts = runner.run_region_inference(
        padded, positions, model, torch.device("cpu"), 256, 1, channel_indices=[0]
    )
    h, w = he.shape[:2]
    return (accum / counts)[0, :h, :w]


def run(args: argparse.Namespace) -> dict:
    data, identity = inputs(args)
    runner = load_code(args.runner, "gigatime_control_runner")
    qc = load_code(args.qc, "gigatime_control_qc")
    if not hasattr(runner, "load_model") or not hasattr(qc, "NuclearAlignmentScore"):
        raise ValueError("The pinned runner or QC API changed")
    if args.preflight_only:
        return {"status": "preflight_pass", "device": "CPU", **identity}
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model, provenance = runner.load_model(
        "prov-gigatime/GigaTIME", None, torch.device("cpu"), True, revision=REVISION
    )
    checkpoint = provenance["checkpoints"][0]
    if checkpoint["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Pinned GigaTIME checkpoint changed")
    args.output.mkdir(parents=True)
    he, predicted = official_inference(data["official_he"], model, runner)
    labels = cv2.resize(data["official_labels"], (512, 512), interpolation=cv2.INTER_NEAREST)
    official_score = score(predicted, labels, qc,
                           "author_paired_measured_DAPI_nuclei")
    preview(he, predicted, labels, args.output / "official-paired-dapi.png")
    results = {"official_paired": official_score}
    for name in ("monusac_original", "monusac_normalized"):
        image = data[name]
        prediction = monusac_inference(image, model, runner)
        results[name] = score(prediction, data["monusac_stardist"], qc,
                              "independent_HE_StarDist_nuclei")
        preview(image, prediction, data["monusac_stardist"], args.output / f"{name}-dapi.png")
    report = {"status": "diagnostic_complete", "device": "CPU",
              "checkpoint_sha256": CHECKPOINT_SHA256, "inputs": identity,
              "results": results,
              "limits": "One author-supplied paired patch and one MoNuSAC field only. This does not validate all GigaTIME proteins, all images, a stain correction, or cell typing; no QC gate is modified."}
    staged = args.output / "REPORT.tmp.json"
    staged.write_text(json.dumps(report, indent=2) + "\n")
    staged.replace(args.output / "REPORT.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-he", type=Path, required=True)
    parser.add_argument("--official-labels", type=Path, required=True)
    parser.add_argument("--monusac-original", type=Path, required=True)
    parser.add_argument("--monusac-normalized", type=Path, required=True)
    parser.add_argument("--stardist", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parsed = parser.parse_args()
    result = run(parsed)
    print(json.dumps({"status": result["status"], "device": "CPU"}))
