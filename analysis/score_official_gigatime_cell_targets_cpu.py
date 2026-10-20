#!/usr/bin/env python3
"""CPU-only five-patch positive control using authors' cell-level targets.

Uses the authors' 556-to-512 resize and four 256-pixel inference windows. This
does not assert that a source-domain test-patch score transfers to ORION.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from skimage.measure import label, regionprops

from validate_orion_nine_raw_case import CHANNELS

ARCHIVE_SHA256 = "cd1212b361f7bfe4f7f01779a59f437c691fd43a684d7b5f584c37bf7d0e2601"
CHECKPOINT_SHA256 = "d200308302c7fefd13b259c62de546f2c859c28d1079e9f67dc8808915d95d0c"
REVISION = "a6556030bdb8071857559e12b921620aa1bbac5b"
PAIR_IDS = ("0_556_556_556", "0_6116_556_556", "0_10564_556_556",
            "0_16680_556_556", "0_30024_556_556")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_runner(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("control_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned CellPhenotyper runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pair(archive: zipfile.ZipFile, ident: str):
    stem = f"sample_test_data/data/{ident}"
    with Image.open(io.BytesIO(archive.read(stem + "_he.png"))) as handle:
        he = np.asarray(handle.convert("RGB"))
    # The pickle is read only from the pinned author-published ZIP SHA-256.
    labels = pickle.loads(gzip.decompress(archive.read(stem + "_comet_binary_thres_labels.pkl.gz")))
    binary = np.unpackbits(np.asarray(labels["comet_array_binary"]), axis=-1)[..., :23]
    if he.shape != (556, 556, 3) or binary.shape != (556, 556, 23) or \
       not np.isin(binary, (0, 1)).all():
        raise ValueError(f"Published paired control shape invalid: {ident}")
    nuclei = np.asarray(labels["labels_dapi"])
    expanded = np.asarray(labels["labels_dapi_expanded"])
    if nuclei.shape != (556, 556) or expanded.shape != (556, 556):
        raise ValueError("Author's nuclear/expanded masks changed")
    return he, binary, nuclei, expanded


def author_cell_targets(binary: np.ndarray, nuclei: np.ndarray,
                        expanded: np.ndarray) -> np.ndarray:
    """Mirror official test notebook mask_noncell=True, cell_mask_label=True.

    The notebook's exact channel spelling is retained, including its TRITIC
    spelling and Ki67 name comparison, to reproduce the published path.
    """
    nuclear_names = {"DAPI", "TRITC", "Cy5", "Ki67_1:150 - TRITC"}
    selected_cell_masks = np.stack(
        [nuclei if ch in nuclear_names else expanded for ch in CHANNELS], axis=-1)
    masked = binary.copy()
    masked[selected_cell_masks == 0] = 0
    processed = np.zeros_like(masked)
    notebook_nuclear_names = {"DAPI", "TRITIC", "Cy5", "Ki67_1:150 - TRITC"}
    for mode, labels in (("nuclei", nuclei), ("cell", expanded)):
        for region in regionprops(label(labels)):
            y0, x0, y1, x1 = region.bbox
            convex = region.convex_image
            ratios = masked[y0:y1, x0:x1][convex].sum(axis=0) / convex.sum()
            for channel, ratio in enumerate(ratios):
                name = CHANNELS[channel]
                valid = name in notebook_nuclear_names if mode == "nuclei" else name not in notebook_nuclear_names
                threshold = .2 if name == "Ki67_1:150 - TRITC" else .5
                if valid and ratio > threshold:
                    view = processed[y0:y1, x0:x1, channel]
                    view[convex] = 1
    if not np.isin(processed, (0, 1)).all():
        raise ValueError("Processed author target is not binary")
    return processed


def predict(he: np.ndarray, model, runner) -> np.ndarray:
    resized = cv2.resize(he, (512, 512), interpolation=cv2.INTER_LINEAR)
    output = np.empty((23, 512, 512), dtype=np.float32)
    with torch.no_grad():
        for y in (0, 256):
            for x in (0, 256):
                patch = resized[y:y+256, x:x+256]
                tensor = runner.normalize_patch_batch(patch[None])
                logits = model(tensor)
                output[:, y:y+256, x:x+256] = torch.sigmoid(logits)[0].numpy()
    return output


def sheet(he: np.ndarray, truth: np.ndarray, pred: np.ndarray,
          rows: list[dict], group: int, path: Path) -> None:
    indices = list(range(group*6, min((group+1)*6, 23)))
    thumb = 192
    image = Image.new("RGB", (4*thumb, len(indices)*(thumb+28)), "white")
    pen = ImageDraw.Draw(image)
    for row, index in enumerate(indices):
        y = row*(thumb+28)
        vals = [he, np.repeat((truth[index]*255)[..., None], 3, axis=2),
                np.repeat(np.rint(np.clip(pred[index], 0, 1)*255).astype(np.uint8)[..., None], 3, axis=2)]
        overlay = np.zeros((512, 512, 3), np.uint8)
        overlay[..., 0] = np.rint(pred[index]*255).astype(np.uint8)
        overlay[..., 1] = truth[index]*255
        vals.append(overlay)
        for col, item in enumerate(vals):
            tile = Image.fromarray(item, "RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
            image.paste(tile, (col*thumb, y+28))
        rec = rows[index]
        auc = rec["roc_auc"]
        label = f"{CHANNELS[index]}  AUC={auc:.3f}" if auc is not None else f"{CHANNELS[index]}  one class"
        pen.text((8, y+6), label, fill="black")
    image.save(path, "WEBP", quality=88)


def run(archive_path: Path, runner_path: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Official paired control must hide all GPUs")
    if output.exists():
        raise FileExistsError(output)
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Author-published paired test archive changed")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    runner = load_runner(runner_path)
    model, provenance = runner.load_model(
        "prov-gigatime/GigaTIME", None, torch.device("cpu"), True, revision=REVISION)
    if provenance["checkpoints"][0]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Pinned GigaTIME CNN checkpoint changed")
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for ident in PAIR_IDS:
            he, binary, nuclei, expanded = load_pair(archive, ident)
            score = predict(he, model, runner)
            prepared = author_cell_targets(binary, nuclei, expanded)
            target = np.stack([cv2.resize(prepared[..., i], (512, 512),
                                          interpolation=cv2.INTER_NEAREST)
                               for i in range(23)])
            resized_he = cv2.resize(he, (512, 512), interpolation=cv2.INTER_LINEAR)
            per_case = []
            for index, channel in enumerate(CHANNELS):
                label = target[index].ravel()
                value = score[index].ravel()
                positives = int(np.sum(label))
                n = len(label)
                record = {"pair": ident, "channel": channel, "target_prevalence": positives/n,
                          "positive_pixels": positives, "roc_auc": None, "average_precision": None}
                if 50 <= positives <= n-50:
                    record["roc_auc"] = float(roc_auc_score(label, value))
                    record["average_precision"] = float(average_precision_score(label, value))
                per_case.append(record)
                rows.append(record)
            for group in range(4):
                sheet(resized_he, target, score, per_case, group,
                      output / f"{ident}-channels-{group+1}.webp")
            temporary = output / f"{ident}.json.tmp"
            temporary.write_text(json.dumps(per_case, indent=2)+"\n")
            temporary.replace(output / f"{ident}.json")
    summary = []
    for channel in CHANNELS:
        valid = [r for r in rows if r["channel"] == channel and r["roc_auc"] is not None]
        summary.append({"channel": channel, "eligible_patches": len(valid),
                        "median_auc": float(np.median([r["roc_auc"] for r in valid])) if valid else None,
                        "median_average_precision": float(np.median([r["average_precision"] for r in valid])) if valid else None,
                        "median_target_prevalence": float(np.median([r["target_prevalence"] for r in valid])) if valid else None})
    report = {"status": "source_domain_paired_positive_control_complete_pending_visual_review",
              "device": "CPU", "source": "Official GigaTIME sample_test_data.zip, five fixed patches",
              "archive_sha256": ARCHIVE_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
              "pair_ids": list(PAIR_IDS), "target": "Authors' binary COMET labels after mask_noncell and cell_mask_label transformations from their public testing notebook, in official 23-channel order",
              "input_contract": "556-pixel source resized to 512; four nonoverlapping 256-pixel model windows",
              "rows": rows, "summary": summary,
              "limit": "Five selected source-domain patches, possibly from the same specimen; not independent external ORION validation. AUROC/AP assess marker-positive localization against the model's binary training target, not raw fluorescence intensity."}
    temporary = output / "REPORT.json.tmp"
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(output / "REPORT.json")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = run(a.archive, a.runner, a.output)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
