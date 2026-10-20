#!/usr/bin/env python3
"""One fixed, CPU-only HEMIT/GigaTIME external-transfer feasibility case."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import tifffile
import torch

from audit_inputs import sha256


CHECKPOINT_SHA256 = "d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c"
REVISION = "a6556030bdb8071857559e12b921620aa1bbac5b"
FIRST_TEST_ID = "[16705,49971]_patch_4_4.tif"
SOURCE_MPP = 0.25  # HEMIT calibration reported by MIPHEI-ViT; source TIFF lacks tags.
TARGETS = (("Pan-CK", "CK", 0), ("CD3", "CD3", 1), ("DAPI", "DAPI", 2))


def load_runner(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("hemit_cellphenotyper_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned CellPhenotyper runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def show_channel(image: np.ndarray, *, probability: bool) -> Image.Image:
    if probability:
        output = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        lo, hi = np.percentile(image, (1, 99.5))
        output = np.rint(np.clip((image - lo) * 255 / max(hi - lo, 1), 0, 255)).astype(np.uint8)
    return Image.fromarray(output).convert("RGB")


def render(rgb: np.ndarray, measured: np.ndarray, scores: np.ndarray,
           rows: list[dict], path: Path) -> None:
    thumb = 256
    columns = ("H&E", "Measured composite", "Measured", "GigaTIME")
    canvas = Image.new("RGB", (len(columns) * thumb, 3 * (thumb + 34) + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for c, title in enumerate(columns):
        draw.text((c * thumb + 8, 8), title, fill="black")
    for row, (measured_name, _, band) in enumerate(TARGETS):
        top = 32 + row * (thumb + 34)
        result = rows[row]
        suffix = f"  AUROC={result['auroc']:.3f}" if result["auroc"] is not None else ""
        draw.text((8, top), f"{measured_name}  rho={result['spearman_rho']:+.3f}{suffix}", fill="black")
        images = [Image.fromarray(rgb, "RGB"), Image.fromarray(measured, "RGB"),
                  show_channel(measured[..., band], probability=False),
                  show_channel(scores[row], probability=True)]
        for c, image in enumerate(images):
            canvas.paste(ImageOps.contain(image, (thumb, thumb), Image.Resampling.LANCZOS),
                         (c * thumb, top + 24))
    canvas.save(path, "WEBP", quality=90)


def run(root: Path, runner_path: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This pilot must hide GPUs with CUDA_VISIBLE_DEVICES=-1")
    if output.exists():
        raise FileExistsError(output)
    input_path = root / "test" / "input" / FIRST_TEST_ID
    label_path = root / "test" / "label" / FIRST_TEST_ID
    rgb = tifffile.imread(input_path)
    measured = tifffile.imread(label_path)
    if rgb.shape != (1024, 1024, 3) or measured.shape != rgb.shape or \
       rgb.dtype != np.uint8 or measured.dtype != np.uint8:
        raise ValueError("Unexpected HEMIT image/label dimensions or byte depth")
    if SOURCE_MPP != 0.25:
        raise ValueError("Changed published HEMIT physical scale")
    runner = load_runner(runner_path)
    torch.set_num_threads(4)
    began = time.monotonic()
    model, provenance = runner.load_model(
        "prov-gigatime/GigaTIME", None, torch.device("cpu"), True, revision=REVISION)
    if provenance["checkpoints"][0]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Pinned GigaTIME CNN checkpoint changed")
    indices = [runner.CHANNEL_NAMES.index(model_name) for _, model_name, _ in TARGETS]
    starts = runner.axis_patch_starts(1024, 256, 128)
    positions = [(y, x) for y in starts for x in starts]
    accum, weights = runner.run_region_inference(
        rgb, positions, model, torch.device("cpu"), 256, 2,
        channel_indices=indices)
    scores = accum / weights
    if scores.shape != (3, 1024, 1024) or not np.isfinite(scores).all():
        raise ValueError("Invalid GigaTIME probability map")
    tissue = np.mean(rgb, axis=-1) < 245
    if tissue.sum() < 100_000:
        raise ValueError("Insufficient tissue for pilot")
    rows = []
    for index, (measured_name, virtual_name, band) in enumerate(TARGETS):
        target = measured[..., band][tissue]
        pred = scores[index][tissue]
        rho = float(spearmanr(target, pred).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"Non-finite correlation for {measured_name}")
        positive = target > 0
        auc = ap = None
        if measured_name != "DAPI" and 50 < positive.sum() < len(positive)-50:
            auc = float(roc_auc_score(positive, pred))
            ap = float(average_precision_score(positive, pred))
        rows.append({"measured": measured_name, "virtual": virtual_name,
                     "label_band": band, "tissue_pixels": int(len(pred)),
                     "positive_prevalence_gt_zero": float(positive.mean()),
                     "spearman_rho": rho, "auroc": auc, "average_precision": ap})
    output.mkdir(parents=True)
    np.savez_compressed(output / "three_marker_probabilities.npz", scores=scores.astype(np.float16))
    render(rgb, measured, scores, rows, output / "paired-contact.webp")
    report = {
        "status": "one_case_exploratory_cpu_pilot_pending_visual_review",
        "device": "CPU", "dataset": "HEMIT official test split", "pair": FIRST_TEST_ID,
        "source_mpp_um": SOURCE_MPP,
        "source_mpp_evidence": "MIPHEI-ViT reports HEMIT at 0.25 um/pixel; released TIFF tags do not encode mpp",
        "model_mpp_um": 0.25, "model_revision": REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runner_sha256": sha256(runner_path),
        "input_sha256": sha256(input_path), "label_sha256": sha256(label_path),
        "tiling": {"patch_px": 256, "stride_px": 128, "windows": len(positions),
                   "weighted_overlap": "CellPhenotyper raised-cosine; divide by accumulated weights"},
        "seconds_including_model_load": round(time.monotonic() - began, 3),
        "rows": rows,
        "limitations": "One deliberately fixed test patch, exploratory only. Label >0 is a provisional Pan-CK/CD3 positivity rule; it needs dataset-level review. Raw mIHC intensity and GigaTIME binary probability are different quantities. No patient-level inference, full-set accuracy, or cell-type prediction is claimed.",
    }
    (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.root, a.runner, a.output)
    print(json.dumps({"status": result["status"], "rows": result["rows"]}, indent=2))
