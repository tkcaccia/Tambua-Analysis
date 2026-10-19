#!/usr/bin/env python3
"""Render registered, cell-level ORION measured/virtual marker comparisons.

The measured map is reconstructed from the released ORION per-cell table and
published cell mask. It is deliberately *not* labelled as raw fluorescence.
All intensity panels use within-field percentile ranks, matching the spatial
ordering evaluated by Spearman rho. No model inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr


def tables(root: Path, patient: str) -> tuple[Path, dict[str, pd.DataFrame], pd.DataFrame]:
    prep = root / "cpu-prep" / patient
    case = root / "gpu-output" / patient / "quantification"
    measured = pd.read_csv(prep / "measured_cells.csv").set_index("CellID", verify_integrity=True)
    frames = {}
    for compartment in ("cell", "cyto", "nuclei"):
        paths = list(case.glob(f"*_{compartment}_gigatime_intensity_stats.csv"))
        if len(paths) != 1:
            raise ValueError(f"Missing or duplicate {compartment} table: {patient}")
        frames[compartment] = pd.read_csv(paths[0]).set_index("label_id", verify_integrity=True)
    return prep, frames, measured


def percentile_colors(values: np.ndarray) -> np.ndarray:
    """One fixed blue-to-yellow palette for ranks; background is handled apart."""
    if not np.isfinite(values).all():
        raise ValueError("Non-finite marker values")
    q = (rankdata(values, method="average") - .5) / len(values)
    points = np.asarray([[37, 54, 97], [23, 123, 150], [69, 191, 136], [249, 218, 93]], float)
    at = np.array([0, 1/3, 2/3, 1], float)
    return np.column_stack([np.interp(q, at, points[:, i]) for i in range(3)]).astype(np.uint8)


def map_cells(mask: np.ndarray, ids: np.ndarray, values: np.ndarray) -> Image.Image:
    # 2048/4 = 512, exact nearest-neighbor spatial sampling, no interpolation.
    small = mask[::4, ::4]
    ix = np.searchsorted(ids, small)
    hit = (ix < len(ids)) & (ids[np.minimum(ix, len(ids)-1)] == small) & (small > 0)
    out = np.full((*small.shape, 3), (236, 240, 239), dtype=np.uint8)
    palette = percentile_colors(values)
    out[hit] = palette[ix[hit]]
    return Image.fromarray(out, "RGB")


def compose(he: Image.Image, mapped: list[Image.Image], patient: str,
            marker: str, target: str, method: str, rho0: float, rho1: float) -> Image.Image:
    width, height, top = 512, 512, 82
    out = Image.new("RGB", (4 * width, height + top + 28), "#f4f8f8")
    draw = ImageDraw.Draw(out)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 19)
    detail_font = ImageFont.truetype(font_path, 15)
    labels = ["Registered H&E", f"Measured {target}", f"Virtual {marker} · mean", f"Virtual {marker} · selected"]
    sub = ["0.325 um/pixel source", "ORION published cell table", f"rho = {rho0:+.3f}", f"rho = {rho1:+.3f}; {method.replace('_', ' ')}"]
    for n, panel in enumerate([he] + mapped):
        x = n * width
        out.paste(panel, (x, top))
        draw.text((x + 12, 10), labels[n], fill="#12313d", font=title_font)
        draw.text((x + 12, 43), sub[n], fill="#42636c", font=detail_font)
    draw.text((13, height + top + 4), f"{patient} | cell-level ranks | blue low / yellow high | grey excluded/unmatched", fill="#45616b", font=detail_font)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    score_rows = list(csv.DictReader(args.scores.open()))
    records = []
    for record in summary["patient_records"]:
        if record["split"] != "evaluation" or not record["technical_qc_passed"]:
            continue
        patient = record["patient"]
        prep, frames, measured = tables(args.root, patient)
        ids = np.asarray(sorted(set(measured.index).intersection(*(set(f.index) for f in frames.values()))), np.int64)
        if len(ids) != record["common_cells"]:
            raise ValueError(f"Common cell mismatch: {patient}")
        measured = measured.loc[ids]
        cell, cyto, nuclei = (frames[name].loc[ids] for name in ("cell", "cyto", "nuclei"))
        centroid = cell[["centroid_y_px", "centroid_x_px"]].to_numpy(float)
        _, neighbors = cKDTree(centroid).query(centroid, k=min(65, len(ids)), workers=1)
        focal = np.arange(len(ids))[:, None]
        neighbors = neighbors[neighbors != focal].reshape(len(ids), -1)
        mask = tifffile.imread(prep / "cells.ome.tif")
        he = Image.fromarray(tifffile.imread(prep / "he.ome.tif").astype(np.uint8), "RGB")
        if mask.shape != (2048, 2048) or he.size != (2048, 2048):
            raise ValueError(f"Unexpected registered image geometry: {patient}")
        he.thumbnail((512, 512), Image.Resampling.LANCZOS)
        for marker, choice in summary["development_selection"].items():
            method = choice["method"]
            pair = next((r for r in score_rows if r["patient"] == patient and r["virtual"] == marker and r["method"] == method), None)
            base = next((r for r in score_rows if r["patient"] == patient and r["virtual"] == marker and r["method"] == "whole_cell_mean"), None)
            if pair is None or base is None:
                raise ValueError(f"Missing frozen correlation: {patient}/{marker}")
            target = pair["measured"]
            truth = pd.to_numeric(measured[target], errors="coerce").to_numpy(float)
            baseline = cell[f"{marker}__mean"].to_numpy(float)
            if method == "whole_cell_mean":
                selected = baseline
            elif method == "whole_cell_maximum":
                selected = cell[f"{marker}__max"].to_numpy(float)
            elif method == "cytoplasm_mean":
                selected = cyto[f"{marker}__mean"].to_numpy(float)
            elif method == "cytoplasm_maximum":
                selected = cyto[f"{marker}__max"].to_numpy(float)
            elif method == "nucleus_mean":
                selected = nuclei[f"{marker}__mean"].to_numpy(float)
            elif method == "nucleus_maximum":
                selected = nuclei[f"{marker}__max"].to_numpy(float)
            elif method == "nucleus_minus_cytoplasm":
                selected = nuclei[f"{marker}__mean"].to_numpy(float) - cyto[f"{marker}__mean"].to_numpy(float)
            elif method == "cytoplasm_minus_nucleus":
                selected = cyto[f"{marker}__mean"].to_numpy(float) - nuclei[f"{marker}__mean"].to_numpy(float)
            elif method.endswith("_local_contrast"):
                source = {"whole_cell_local_contrast": cell, "cytoplasm_local_contrast": cyto,
                          "nucleus_local_contrast": nuclei}[method]
                vector = source[f"{marker}__mean"].to_numpy(float)
                selected = vector - np.median(vector[neighbors], axis=1)
            else:
                raise ValueError(f"Unknown selected summary: {method}")
            finite = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(selected)
            rho0 = float(spearmanr(truth[finite], baseline[finite]).statistic)
            rho1 = float(spearmanr(truth[finite], selected[finite]).statistic)
            if abs(rho0 - float(base["spearman_rho"])) > 1e-8 or abs(rho1 - float(pair["spearman_rho"])) > 1e-8:
                raise ValueError(f"Frozen score mismatch: {patient}/{marker}: {rho0}, {rho1}")
            common_ids = ids[finite]
            panels = [map_cells(mask, common_ids, v[finite]) for v in (truth, baseline, selected)]
            image = compose(he, panels, patient, marker, target, method, rho0, rho1)
            filename = f"{patient}-{marker.replace('/', '_')}.webp"
            image.save(args.out / filename, "WEBP", quality=82, method=5)
            records.append({"patient": patient, "marker": marker, "measured": target, "method": method,
                            "baseline_rho": rho0, "selected_rho": rho1, "delta_rho": rho1-rho0,
                            "common_cells": int(finite.sum()), "qc": "DAPI nuclear and seam pass",
                            "image": filename})
        print(f"Rendered {patient}: {len(summary['development_selection'])} markers", flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"scope": "10 reserved ORION patients with passing DAPI and seam QC; published whole-cell measurements rendered on published cell IDs; rank display only",
                                                        "records": records}, indent=2) + "\n")
    print(f"Rendered {len(records)} patient-marker comparisons")


if __name__ == "__main__":
    main()
