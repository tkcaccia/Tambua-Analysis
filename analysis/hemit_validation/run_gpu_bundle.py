#!/usr/bin/env python3
"""Fail-fast, checkpointed HEMIT external validation on Chiamaka's single GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import tifffile
import torch

from audit_inputs import sha256
from pilot_cpu import (CHECKPOINT_SHA256, REVISION, SOURCE_MPP, TARGETS,
                       load_runner, render)


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def process_startticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    return int(text.rsplit(")", 1)[1].split()[19])


def score_image(measured: np.ndarray, scores: np.ndarray,
                tissue: np.ndarray) -> list[dict]:
    rows = []
    for index, (measured_name, virtual_name, band) in enumerate(TARGETS):
        target = measured[..., band][tissue]
        pred = scores[index][tissue]
        if len(pred) < 100_000:
            raise ValueError("Insufficient tissue pixels")
        rho = float(spearmanr(target, pred).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"Non-finite {measured_name} Spearman")
        positive = target > 0
        auc = ap = None
        if measured_name != "DAPI" and 50 < positive.sum() < len(positive) - 50:
            auc = float(roc_auc_score(positive, pred))
            ap = float(average_precision_score(positive, pred))
        rows.append({"measured": measured_name, "virtual": virtual_name,
                     "label_band": band, "tissue_pixels": int(len(pred)),
                     "positive_prevalence_gt_zero": float(positive.mean()),
                     "spearman_rho": rho, "auroc": auc, "average_precision": ap})
    return rows


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This Chiamaka bundle is scoped to CUDA_VISIBLE_DEVICES=0")
    if args.output.exists():
        raise FileExistsError(args.output)
    if abs(SOURCE_MPP - 0.25) > 1e-9:
        raise ValueError("HEMIT published scale no longer matches model grid")
    audit = json.loads(args.audit.read_text())
    entries = audit["inspected_pairs"]
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if len(ids) != 292 or len(entries) != 292 or \
       [entry["id"] for entry in entries] != ids or \
       audit["published_ids_sha256"] != sha256(args.ids):
        raise ValueError("Frozen 292-case HEMIT audit/selection mismatch")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()
    if len(gpu) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly Chiamaka's single assigned GPU")
    args.output.mkdir(parents=True)
    launch = {
        "status": "running", "host": socket.gethostname(),
        "pid": os.getpid(), "proc_startticks": process_startticks(os.getpid()),
        "process_group_id": os.getpgrp(), "session_id": os.getsid(0),
        "gpu": gpu[0], "cuda_visible_devices": "0",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_mpp_um": SOURCE_MPP,
        "source_mpp_evidence": "MIPHEI-ViT HEMIT=0.25 um/pixel; source TIFF lacks pixel-size tags",
        "model_mpp_um": 0.25, "cases": len(entries),
        "audit_sha256": sha256(args.audit), "ids_sha256": sha256(args.ids),
        "runner_sha256": sha256(args.runner),
        "script_sha256": sha256(Path(__file__)),
    }
    atomic_json(args.output / "BUNDLE_STARTED.json", launch)
    torch.set_num_threads(4)
    runner = load_runner(args.runner)
    model, provenance = runner.load_model(
        "prov-gigatime/GigaTIME", None, torch.device("cuda:0"), True, revision=REVISION)
    if provenance["checkpoints"][0]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Pinned GigaTIME checkpoint changed")
    from gigatime_seam_qc import compute_seam_metrics

    channel_indices = [runner.CHANNEL_NAMES.index(model_name) for _, model_name, _ in TARGETS]
    summaries = []
    for index, entry in enumerate(entries, start=1):
        name = entry["id"]
        case = args.output / f"case-{index:03d}"
        case.mkdir()
        atomic_json(case / "CASE_STARTED.json", {"index": index, "id": name,
                   "started_utc": datetime.now(timezone.utc).isoformat()})
        input_path = args.root / "test" / "input" / name
        label_path = args.root / "test" / "label" / name
        if sha256(input_path) != entry["input_sha256"] or \
           sha256(label_path) != entry["label_sha256"]:
            raise ValueError(f"Frozen pair checksum changed: {name}")
        rgb = tifffile.imread(input_path)
        measured = tifffile.imread(label_path)
        if rgb.shape != (1024, 1024, 3) or measured.shape != rgb.shape or \
           rgb.dtype != np.uint8 or measured.dtype != np.uint8:
            raise ValueError(f"Unexpected HEMIT TIFF geometry or byte depth: {name}")
        start = time.monotonic()
        accum, weights, shape = runner.run_inference(
            rgb, model, torch.device("cuda:0"), 256, 128, 2, False, case)
        torch.cuda.synchronize()
        if shape != (1024, 1024) or np.any(weights <= 0):
            raise RuntimeError(f"Invalid weighted GigaTIME output: {name}")
        scores = (accum[channel_indices] / weights).astype(np.float32)
        del accum, weights
        if scores.shape != (3, 1024, 1024) or not np.isfinite(scores).all():
            raise RuntimeError(f"Non-finite GigaTIME output: {name}")
        seam = compute_seam_metrics(scores, block_size=256,
                                    channel_names=[t[1] for t in TARGETS])
        atomic_json(case / "SEAM_QC.json", seam)
        if seam["status"] != "pass":
            raise RuntimeError(f"Technical seam QC failed: {name}")
        tissue = np.mean(rgb, axis=-1) < 245
        rows = score_image(measured, scores, tissue)
        temporary = case / "three_marker_probabilities.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, scores=scores.astype(np.float16))
        temporary.replace(case / "three_marker_probabilities.npz")
        render(rgb, measured, scores, rows, case / "paired-contact.webp")
        result = {"status": "case_complete_pending_visual_review",
                  "index": index, "id": name,
                  "input_sha256": entry["input_sha256"],
                  "label_sha256": entry["label_sha256"],
                  "prediction_sha256": sha256(case / "three_marker_probabilities.npz"),
                  "seconds": round(time.monotonic() - start, 3),
                  "seam_qc_status": seam["status"], "rows": rows}
        atomic_json(case / "CASE_COMPLETE.json", result)
        summaries.append(result)
        print(f"[{index:03d}/292] {name} complete {result['seconds']:.2f}s", flush=True)
        del scores, rgb, measured
        torch.cuda.empty_cache()
    report = {"status": "all_292_gpu_cases_complete_pending_visual_review",
              "started": launch, "model_revision": REVISION,
              "checkpoint_sha256": CHECKPOINT_SHA256,
              "cases": [{k: r[k] for k in ("index", "id", "seconds", "seam_qc_status", "rows")}
                        for r in summaries]}
    atomic_json(args.output / "BUNDLE_COMPLETE.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "count": len(result["cases"])}, indent=2))
