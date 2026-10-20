#!/usr/bin/env python3
"""Bounded, resumable CPU-only GigaTIME inference for the 93-image expansion.

No GPU is visible or requested. A technical execution failure stops the entire
bundle; a scientific nuclear-QC failure is recorded and excludes that image
from later cell typing without stopping inference of other images.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import zarr


REVISION = "a6556030bdb8071857559e12b921620aa1bbac5b"
CHANNELS = ("DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34",
            "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1",
            "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B",
            "Transgelin")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(obj, indent=2) + "\n")
    staged.replace(path)


def bounded(command: list[str], log: Path, seconds: int, env: dict) -> float:
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise RuntimeError(f"Timed out after {seconds}s: {' '.join(command[:2])}")
    if code:
        raise RuntimeError(f"Exit {code}; inspect {log}")
    return time.monotonic() - start


def validate_case(case: Path, source: dict, image_sha: str) -> dict:
    output = case / "gigatime_output"
    store = zarr.open(str(output / "gigatime_probs.zarr"), mode="r")
    attrs = store.attrs["gigatime"]
    if tuple(attrs["store_channels"]) != CHANNELS or \
            abs(attrs["source_mpp"] - float(source["mpp"])) > 1e-6 or \
            abs(attrs["effective_mpp"] - .25) > 1e-6 or \
            store["0"].shape[0] != 23:
        raise ValueError("Channel identity or physical-resolution contract failed")
    seam = json.loads((output / "gigatime_seam_qc.json").read_text())
    qc = json.loads((case / "stardist_qc.json").read_text())
    if seam["status"] != "pass" or qc["status"] not in ("pass", "fail") or \
            qc["nuclei_mask"] != str((case / "stardist_labels.tif").resolve()):
        raise ValueError("Seam or independent StarDist nuclear QC record failed")
    return {"sample_id": source["sample_id"], "patient_id": source["patient_id"],
            "organ": source["organ"], "source_mpp": float(source["mpp"]),
            "model_mpp": attrs["effective_mpp"], "source_rgb_sha256": image_sha,
            "channels": list(CHANNELS), "image_shape_yx": attrs["inference_shape_yx"],
            "nuclear_qc": {name: qc[name] for name in
                           ("status", "alignment_status", "border_shadow_status",
                            "spatial_shadow_status", "auc", "nuclear_boundary_contrast")},
            "seam_qc": seam["status"]}


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU bundle requires CUDA_VISIBLE_DEVICES=-1")
    os.nice(10)
    prepared_marker = args.prepared / "PREP_COMPLETE.json"
    prepared = json.loads(prepared_marker.read_text())
    if prepared["cases"] != 93 or not prepared["all_scanner_rgb"]:
        raise ValueError("Frozen scanner-RGB expansion changed")
    source = {row["sample_id"]: row for row in prepared["records"]}
    match = [row for folder in (args.fit_match, args.eval_match)
             for row in json.loads((folder / "SUMMARY.json").read_text())["records"]
             if row["sample_id"] in source]
    if len(source) != 93 or len(match) != 93 or \
            {row["sample_id"] for row in match} != set(source):
        raise ValueError("Prepared and detected expansion sets disagree")
    fit_ids = {row["sample_id"] for row in match if row["split"] == "train"}
    if len(fit_ids) != 44:
        raise ValueError("Frozen 44-fit/49-evaluation partition changed")
    # Alternate fit and evaluation so an interrupted CPU run yields diagnostic
    # observations from both partitions, without using labels to select cases.
    fit_order = sorted(fit_ids)
    eval_order = sorted(set(source) - fit_ids)
    order = [item for pair in zip(fit_order + [None] * (len(eval_order) - len(fit_order)),
                                  eval_order) for item in pair if item is not None]
    if len(order) != 93 or len(set(order)) != 93:
        raise ValueError("CPU case order lost a sample")
    if args.dry_run:
        return {"device": "CPU", "cases": len(order), "first": order[:5]}
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"device": "CPU", "prepared_sha256": sha(prepared_marker),
                "runner_sha256": sha(args.runner), "qc_sha256": sha(args.qc),
                "fit_match_sha256": sha(args.fit_match / "SUMMARY.json"),
                "eval_match_sha256": sha(args.eval_match / "SUMMARY.json"),
                "case_order": order, "model_revision": REVISION}
    identity_path = args.output / "BUNDLE_IDENTITY.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("CPU resume identity differs from frozen bundle")
    else:
        atomic_json(identity_path, identity)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    outcomes = []
    try:
        for index, sample in enumerate(order, 1):
            folder = args.output / sample
            src = args.prepared / "samples" / sample
            image = src / "he.ome.tif"
            if source[sample]["source_color_reference"] != "scanner_svs_rgb":
                raise ValueError(f"Scanner RGB not verified: {sample}")
            image_sha = sha(image)
            marker = folder / "CASE_COMPLETE.json"
            if marker.exists():
                result = json.loads(marker.read_text())
                if result["source_rgb_sha256"] != image_sha:
                    raise ValueError(f"Completed CPU case identity changed: {sample}")
                validate_case(folder, source[sample], image_sha)
                outcomes.append(result)
                print(f"REUSED {index}/93 {sample} QC={result['nuclear_qc']['status']}", flush=True)
                continue
            if folder.exists():
                raise ValueError(f"Partial CPU case requires audit before resume: {sample}")
            if shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB disk reserve; stop CPU bundle")
            folder.mkdir()
            mask_root = args.fit_detected if sample in fit_ids else args.eval_detected
            mask = mask_root / sample / "stardist_labels.tif"
            if not mask.is_file():
                raise FileNotFoundError(mask)
            # A symlink records the exact independent mask used in the QC report.
            (folder / "stardist_labels.tif").symlink_to(mask)
            elapsed = bounded([str(args.python), "-u", str(args.runner),
                               "--image", str(image), "--outdir", str(folder / "gigatime_output"),
                               "--device", "cpu", "--repo-revision", REVISION,
                               "--auto-parameters", "--output-format", "zarr",
                               "--jpg-markers", "DAPI,CK,CD3,CD68,CD14,CD16",
                               "--seam-qc-mode", "warn"], folder / "inference.log", 180, env)
            qc_elapsed = bounded([str(args.python), str(args.qc),
                                  "--image", str(folder / "gigatime_output"),
                                  "--nuclei", str(folder / "stardist_labels.tif"),
                                  "--output", str(folder / "stardist_qc.json"),
                                  "--mode", "warn"], folder / "qc.log", 90, env)
            result = validate_case(folder, source[sample], image_sha)
            result.update(inference_seconds=elapsed, qc_seconds=qc_elapsed)
            atomic_json(marker, result)
            outcomes.append(result)
            print(f"COMPLETE {index}/93 {sample} QC={result['nuclear_qc']['status']} "
                  f"inference={elapsed:.1f}s", flush=True)
    except Exception as exc:
        atomic_json(args.output / "BUNDLE_FAILED.json",
                    {"status": "failed", "completed_cases": len(outcomes),
                     "next_case": order[len(outcomes)] if len(outcomes) < len(order) else None,
                     "error": f"{type(exc).__name__}: {exc}"})
        raise
    summary = {"status": "complete", "device": "CPU", "cases": len(outcomes),
               "nuclear_qc": dict(Counter(row["nuclear_qc"]["status"] for row in outcomes)),
               "case_order": order,
               "limit": "Scientific QC failures are excluded later. No cell quantification or fastPLS score is implied by inference completion."}
    atomic_json(args.output / "BUNDLE_COMPLETE.json", summary)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "fit-match", "eval-match", "fit-detected",
                 "eval-detected", "runner", "qc", "python", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(args)))
