#!/usr/bin/env python3
"""Single-field CPU check of the current pinned GigaTIME inference code.

This bounded diagnostic saves virtual DAPI only. It neither changes the
historical marker tables nor validates virtual cell-type markers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
import torch


def histogram_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    p = np.bincount(np.clip(np.rint(pos * 255), 0, 255).astype(np.uint8), minlength=256)
    n = np.bincount(np.clip(np.rint(neg * 255), 0, 255).astype(np.uint8), minlength=256)
    return float(np.sum(p * (np.cumsum(n) - n + 0.5 * n)) / (pos.size * neg.size))


def score(dapi: np.ndarray, ids: np.ndarray) -> dict:
    if ids.shape != dapi.shape:
        ids = np.asarray(Image.fromarray(ids).resize(
            (dapi.shape[1], dapi.shape[0]), Image.Resampling.NEAREST))
    positive = ids > 0
    negative = binary_dilation(positive, iterations=8) & ~positive
    h, w = dapi.shape
    rows = []
    for r in range(3):
        for c in range(3):
            ys, xs = slice(r * h // 3, (r + 1) * h // 3), slice(c * w // 3, (c + 1) * w // 3)
            values, pos, neg = dapi[ys, xs], positive[ys, xs], negative[ys, xs]
            rows.append({"row": r, "column": c, "auc": histogram_auc(values[pos], values[neg])
                         if pos.sum() >= 100 and neg.sum() >= 100 else None})
    edge = np.zeros((h, w), bool)
    edge[:24] = edge[-24:] = True
    edge[:, :24] = edge[:, -24:] = True
    return {"nuclear_nearby_auc": histogram_auc(dapi[positive], dapi[negative]),
            "local_auc": rows,
            "edge_to_core_nuclear_mean_ratio": float(dapi[positive & edge].mean() /
                                                     dapi[positive & ~edge].mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--nuclei", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code / "bin"))
    from run_gigatime_on_crop import (  # noqa: E402
        GigaTIMEModel, inspect_crop_image, read_crop_image,
        pad_to_tiling_shape, run_region_inference,
    )
    start = time.monotonic()
    torch.set_num_threads(4)
    plan = inspect_crop_image(str(args.image), 0, 100, 4096, .25, 8.,
                              num_output_channels=1, bytes_per_sample=1,
                              strict_target_mpp=True)
    rgb = read_crop_image(str(args.image), 0, plan)
    padded, positions = pad_to_tiling_shape(rgb, 256, 128)
    weights = Path(
        "/home/chiamaka/.cache/huggingface/hub/models--prov-gigatime--GigaTIME/"
        "snapshots/a6556030bdb8071857559e12b921620aa1bbac5b/model.pth"
    )
    model = GigaTIMEModel(num_classes=23, input_channels=3)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
    model.eval()
    accum, counts = run_region_inference(padded, positions, model, torch.device("cpu"),
                                         256, 1, channel_indices=[0])
    dapi = (accum / counts)[0, :rgb.shape[0], :rgb.shape[1]]
    with Image.open(args.nuclei) as source:
        ids = np.asarray(source, dtype=np.int32)
    result = {"image": str(args.image), "nuclei": str(args.nuclei),
              "model_revision": "a6556030bdb8071857559e12b921620aa1bbac5b",
              "device": "cpu", "image_shape_yx": list(dapi.shape),
              "patch_count": len(positions), "elapsed_seconds": time.monotonic() - start,
              "model_grid_mpp": plan["effective_mpp"], "qc": score(dapi, ids)}
    (args.outdir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    gray = np.uint8(np.clip(dapi, 0, 1) * 255)
    Image.fromarray(gray).save(args.outdir / "dapi.png")
    if ids.shape != dapi.shape:
        ids = np.asarray(Image.fromarray(ids).resize(
            (dapi.shape[1], dapi.shape[0]), Image.Resampling.NEAREST))
    nucleus = ids > 0
    outline = nucleus & ~binary_erosion(nucleus)
    overlay = np.repeat(gray[..., None], 3, axis=2)
    overlay[outline] = (255, 45, 45)
    Image.fromarray(overlay).save(args.outdir / "dapi_expert_overlay.png")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
