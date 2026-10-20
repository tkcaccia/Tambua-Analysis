#!/usr/bin/env python3
"""Explore label-free virtual-DAPI illumination correction on train fields.

The untouched source predictions and StarDist masks are used only for QC. No
test field is loaded. The selected transform is diagnostic until raw measured
ORION DAPI and non-DAPI marker specificity have been independently checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr


SPECS = ("raw", "divide_48", "divide_96", "subtract_48", "subtract_96",
         "logit_subtract_48", "logit_subtract_96")


def score_module(path: Path):
    if not path.is_file():
        raise ValueError("Frozen CellPhenotyper nuclear QC module missing")
    sys.path.insert(0, str(path.parent))
    import gigatime_nuclear_qc  # type: ignore
    if Path(gigatime_nuclear_qc.__file__).resolve() != path.resolve():
        raise ValueError("Loaded nuclear QC module differs from requested file")
    return gigatime_nuclear_qc


def correction(raw: np.ndarray, spec: str) -> np.ndarray:
    p = raw.astype(np.float32) / 65535.0
    if spec == "raw":
        return p
    family, width = spec.rsplit("_", 1)
    sigma = float(width)
    if family == "divide":
        background = gaussian_filter(p, sigma=sigma, mode="reflect")
        reference = float(np.median(background))
        return np.clip(p * reference / np.maximum(background, .025), 0, 1)
    if family == "subtract":
        residual = p - gaussian_filter(p, sigma=sigma, mode="reflect")
    elif family == "logit_subtract":
        clipped = np.clip(p, 1e-4, 1 - 1e-4)
        logits = np.log(clipped / (1 - clipped))
        residual = logits - gaussian_filter(logits, sigma=sigma, mode="reflect")
    else:
        raise ValueError(f"Unknown transform: {spec}")
    lo, hi = np.quantile(residual, [.01, .997])
    if hi <= lo:
        raise ValueError("Uninformative corrected image")
    return np.clip((residual-lo)/(hi-lo), 0, 1)


def aligned_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y = np.minimum(labels.shape[0]-1,
                   np.floor((np.arange(shape[0])+.5)*labels.shape[0]/shape[0]).astype(int))
    x = np.minimum(labels.shape[1]-1,
                   np.floor((np.arange(shape[1])+.5)*labels.shape[1]/shape[1]).astype(int))
    return labels[np.ix_(y, x)]


def review_panel(original: np.ndarray, corrected: np.ndarray,
                 labels: np.ndarray, path: Path) -> None:
    side = 320
    panel = Image.new("RGB", (side*2, side+28), "white")
    pen = ImageDraw.Draw(panel)
    pen.text((8, 8), "Saved virtual DAPI", fill="black")
    pen.text((side+8, 8), "Label-free candidate", fill="black")
    from scipy.ndimage import binary_erosion
    edge = (labels>0) & ~binary_erosion(labels>0,iterations=1)
    for index, array in enumerate((original,corrected)):
        gray = np.rint(np.clip(array,0,1)*255).astype(np.uint8)
        rgb = np.repeat(gray[...,None],3,axis=-1)
        rgb[edge] = (255,35,35)
        image=Image.fromarray(rgb)
        image.thumbnail((side,side),Image.Resampling.LANCZOS)
        panel.paste(image,(index*side+(side-image.width)//2,28+(side-image.height)//2))
    panel.save(path,"WEBP",quality=88)


def run(snapshot: Path, qc_code: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Training correction audit must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    frozen=json.loads((snapshot/"SNAPSHOT.json").read_text())
    if frozen["status"]!="training_only_read_only_snapshot" or not frozen["records"] or \
            any(r["split"]!="train" for r in frozen["records"]):
        raise ValueError("Expected only frozen training fields")
    qc=score_module(qc_code)
    output.mkdir(parents=True)
    rows=[]
    samples=[]
    for item in frozen["records"]:
        path=snapshot/f"{item['sample_id']}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["array_sha256"]:
            raise ValueError("Training source array hash changed")
        with np.load(path) as source:
            raw=source["dapi"]
            labels=aligned_labels(source["stardist_source_mask"],raw.shape)
        baseline=correction(raw,"raw")
        for spec in SPECS:
            transformed=correction(raw,spec)
            scorer=qc.NuclearAlignmentScore(raw.shape)
            scorer.add(transformed,labels,0,0)
            result=scorer.report()
            sampled_a=baseline[::scorer.stride,::scorer.stride].ravel()
            sampled_b=transformed[::scorer.stride,::scorer.stride].ravel()
            rho=float(spearmanr(sampled_a,sampled_b).statistic)
            rows.append({"sample_id":item["sample_id"],"patient_id":item["patient_id"],
                         "organ":item["organ"],"spec":spec,"status":result["status"],
                         "auc":result["auc"],"dice":result["prevalence_matched_dice"],
                         "boundary_contrast":result["nuclear_boundary_contrast"],
                         "spatial_shadow_status":result["spatial_shadow_status"],
                         "border_shadow_status":result.get("border_shadow_status"),
                         "rank_rho_to_raw":rho})
        samples.append((item["sample_id"],raw,labels))
    totals=[]
    for spec in SPECS:
        subset=[r for r in rows if r["spec"]==spec]
        aucs=[r["auc"] for r in subset if r["auc"] is not None]
        totals.append({"spec":spec,"cases":len(subset),
                       "pass":sum(r["status"]=="pass" for r in subset),
                       "spatial_shadow_fail":sum(r["spatial_shadow_status"]=="fail" for r in subset),
                       "median_auc":float(np.median(aucs)) if aucs else None,
                       "median_rank_rho_to_raw":float(np.median([r["rank_rho_to_raw"]
                                                                  for r in subset]))})
    # The ranking is determined entirely by training data. Raw wins ties.
    best=max(totals,key=lambda r:(r["pass"],r["median_auc"] or -1,
                                  r["median_rank_rho_to_raw"]))
    case_by_name={r["sample_id"]:r for r in rows if r["spec"]==best["spec"]}
    candidates=[x for x in samples if case_by_name[x[0]]["status"]=="pass"][:6]
    if not candidates:
        candidates=samples[:6]
    for index,(name,raw,labels) in enumerate(candidates,1):
        review_panel(correction(raw,"raw"),correction(raw,best["spec"]),labels,
                     output/f"training-review-{index:02d}.webp")
    report={"status":"training_only_diagnostic","device":"CPU",
            "frozen_training_cases":len(frozen["records"]),
            "source_snapshot_sha256":hashlib.sha256((snapshot/"SNAPSHOT.json").read_bytes()).hexdigest(),
            "qc_code_sha256":hashlib.sha256(qc_code.read_bytes()).hexdigest(),
            "candidate_specs":list(SPECS),"summary":totals,
            "train_selected_spec":best["spec"],"per_image":rows,
            "limit":"Train-only screening of postprocessed DAPI with the unchanged H&E/StarDist gate. No test evaluation, ORION measured-marker validation, or permission to replace saved raw GigaTIME output. A high pass rate alone cannot certify protein specificity or absence of visible shadow."}
    (output/"REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--qc-code",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=run(a.snapshot,a.qc_code,a.output)
    print(json.dumps({"train_cases":r["frozen_training_cases"],
                      "selected":r["train_selected_spec"],"summary":r["summary"]}))
