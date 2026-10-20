#!/usr/bin/env python3
"""Render every QC-passing expanded field and its exploratory cell calls.

This gallery is diagnostic. Technical DAPI/shadow passes do not establish
marker identity, and these fastPLS calls are not validated cell-type results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile
import zarr

from quantify_detected_monusac_cells import GIGATIME_CHANNELS


CLASSES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")
COLORS = {"epithelial": (251, 149, 72), "lymphocyte": (66, 193, 220),
          "macrophage": (188, 126, 224), "neutrophil": (136, 207, 108)}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(image: Image.Image, path: Path) -> None:
    image.save(path, "WEBP", quality=94, method=5)


def overlay(rgb: np.ndarray, ids: np.ndarray, labels: dict[int, str],
            *, errors_against: dict[int, str] | None = None) -> Image.Image:
    if ids.shape != rgb.shape[:2]:
        raise ValueError("Overlay mask is not registered to scanner RGB")
    unique = set(int(value) for value in np.unique(ids) if value)
    if not set(labels) <= unique:
        raise ValueError("Overlay label ID is absent from instance mask")
    palette = np.full((int(ids.max()) + 1, 3), 120, dtype=np.uint8)
    for index, name in labels.items():
        if name not in COLORS:
            raise ValueError("Unknown cell type in gallery")
        palette[index] = (COLORS[name] if errors_against is None else
                          ((63, 188, 113) if errors_against.get(index) == name
                           else (232, 77, 80)))
    mask = ids > 0
    out = rgb.copy()
    out[mask] = np.rint(rgb[mask] * .28 + palette[ids][mask] * .72).astype(np.uint8)
    return Image.fromarray(out)


def display_marker(channel: np.ndarray) -> tuple[Image.Image, Image.Image, list[int]]:
    if channel.dtype != np.uint16 or channel.ndim != 2:
        raise ValueError("Expected native uint16 GigaTIME marker plane")
    raw = np.asarray(channel, dtype=np.float32) / 65535.0
    linear = Image.fromarray(np.rint(raw * 255).astype(np.uint8), mode="L")
    lo, hi = np.quantile(raw, (.01, .997))
    if hi <= lo + 1e-6:
        stretched = np.zeros_like(raw, dtype=np.uint8)
    else:
        stretched = np.rint(np.clip((raw - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return linear, Image.fromarray(stretched, mode="L"), [int(round(lo * 65535)),
                                                           int(round(hi * 65535))]


def contact(root: Path, names: list[str], destination: Path, rows: int,
            cols: int, *, label: str) -> None:
    width, height = 168, 142
    sheet = Image.new("RGB", (cols * width, rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, filename in enumerate(names):
        with Image.open(root / filename) as opened:
            image = opened.convert("RGB")
        image.thumbnail((width - 6, height - 28),
                        getattr(Image, "Resampling", Image).LANCZOS)
        x, y = index % cols * width, index // cols * height
        sheet.paste(image, (x, y + 23))
        draw.text((x + 3, y + 3), f"{index + 1:02d} {label}", fill="black")
    save(sheet, destination)


def build(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort = json.loads((args.quantification / "COHORT_COMPLETE.json").read_text())
    study = json.loads((args.study / "STUDY_COMPLETE.json").read_text())
    if cohort.get("status") != "complete" or cohort.get("qc_passing_images") != 9 or \
            study.get("publication_status") != \
            "exploratory_do_not_publish_as_validated_cell_type_accuracy" or \
            study.get("all_detected_expansion_cells_predicted") != 469:
        raise ValueError("Expected complete, explicitly exploratory nine-field study")
    calls = read_rows(args.study / "all_expansion_detected_predictions.csv")
    predicted = {(row["sample_id"], int(row["label_id"])): row["predicted_cell_type"]
                 for row in calls}
    if len(predicted) != len(calls) or len(calls) != 469:
        raise ValueError("Duplicate or missing expanded detected-cell calls")
    args.output.mkdir(parents=True)
    asset = args.output / "assets"
    asset.mkdir()
    records = []
    total_confusion = np.zeros((4, 4), dtype=np.int64)
    for item in cohort["records"]:
        sample, split = item["sample_id"], item["split"]
        source = args.prepared / "samples" / sample
        quantified = args.quantification / sample
        case = args.inference / sample
        provenance = json.loads((source / "provenance.json").read_text())
        checkpoint = json.loads((quantified / "CASE_COMPLETE.json").read_text())
        qc = json.loads((case / "stardist_qc.json").read_text())
        if provenance.get("source_color_reference") != "scanner_svs_rgb" or \
                checkpoint.get("source_color_reference") != "scanner_svs_rgb" or \
                checkpoint.get("quantifier_sha256") != cohort.get("quantifier_sha256") or \
                any(qc.get(field) != "pass" for field in
                    ("status", "alignment_status", "border_shadow_status",
                     "spatial_shadow_status")):
            raise ValueError(f"Scanner RGB, quantifier or DAPI/shadow QC changed: {sample}")
        rgb = tifffile.imread(source / "he.ome.tif")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Wrong scanner-RGB image layout: {sample}")
        detected_root = args.fit_detected if split == "train" else args.eval_detected
        ids = tifffile.imread(detected_root / sample / "stardist_labels.tif")
        source_ids = tifffile.imread(source / "expert_nuclei.ome.tif")
        if ids.shape != rgb.shape[:2] or source_ids.shape != ids.shape:
            raise ValueError(f"Native mask/image geometry differs: {sample}")
        actual = {int(row["label_id"]): row["cell_type"] for row in
                  read_rows(quantified / "expert_types_on_detected.csv")}
        reference = {int(row["label_id"]): row["cell_type"] for row in
                     read_rows(source / "expert_types.csv")}
        called = {index: value for (case_id, index), value in predicted.items()
                  if case_id == sample}
        detected_ids = set(int(value) for value in np.unique(ids) if value)
        if set(called) != detected_ids or not set(actual) <= detected_ids or \
                checkpoint["detected_cells_quantified"] != len(detected_ids):
            raise ValueError(f"Predictions do not cover every detected nucleus: {sample}")
        if not set(reference) <= set(int(value) for value in np.unique(source_ids)):
            raise ValueError(f"Expert source labels missing from mask: {sample}")
        panels = {
            "he": Image.fromarray(rgb),
            "source-expert": overlay(rgb, source_ids, reference),
            "detected": overlay(rgb, ids,
                                {index: "lymphocyte" for index in detected_ids}),
            "matched-expert": overlay(rgb, ids, actual),
            "predicted": overlay(rgb, ids, called),
            "errors": overlay(rgb, ids, {index: called[index] for index in actual},
                              errors_against=actual),
        }
        for name, image in panels.items():
            save(image, asset / f"{sample}--{name}.webp")
        store = zarr.open(str(case / "gigatime_output/gigatime_probs.zarr"), mode="r")
        marker_meta = store.attrs["gigatime"]
        marker_values = store["0"]
        if marker_meta["channels"] != list(GIGATIME_CHANNELS) or \
                abs(float(marker_meta["effective_mpp"]) - .25) > 1e-6 or \
                abs(float(marker_meta["source_mpp"]) - float(provenance["mpp"])) > 1e-6 or \
                marker_values.shape[0] != 23:
            raise ValueError(f"GigaTIME channel order or physical scale changed: {sample}")
        quantiles = {}
        for index, channel_name in enumerate(GIGATIME_CHANNELS):
            linear, contrast, quantile = display_marker(np.asarray(marker_values[index]))
            save(linear, asset / f"{sample}--marker-{index:02d}-linear.webp")
            save(contrast, asset / f"{sample}--marker-{index:02d}-contrast.webp")
            quantiles[channel_name] = quantile
        confusion = np.zeros((4, 4), dtype=np.int64)
        for cell_id, true_class in actual.items():
            confusion[CLASSES.index(true_class), CLASSES.index(called[cell_id])] += 1
        if split == "test":
            total_confusion += confusion
        support = Counter(actual.values())
        recalls = {name: (float(confusion[i, i] / support[name]) if support[name]
                          else None) for i, name in enumerate(CLASSES)}
        marker_qc = json.loads((case / "gigatime_output/gigatime_marker_score_qc.json").read_text())
        records.append({"sample_id": sample, "patient_id": item["patient_id"],
                        "organ": item["organ"], "split": split,
                        "source_shape_yx": list(ids.shape),
                        "source_mpp": provenance["mpp"], "model_mpp": .25,
                        "model_shape_yx": list(marker_values.shape[1:]),
                        "detected_cells": len(detected_ids),
                        "matched_cells": len(actual),
                        "matched_accuracy": float(np.trace(confusion) / len(actual)),
                        "expert_support": {name: support[name] for name in CLASSES},
                        "expert_recall": recalls,
                        "confusion_matrix": confusion.tolist(),
                        "marker_distribution_qc": marker_qc["status"],
                        "marker_distribution_warnings": marker_qc["warnings"],
                        "marker_display_p1_p997_uint16": quantiles})
    if len(records) != 9 or sum(row["detected_cells"] for row in records) != 469 or \
            total_confusion.tolist() != study["new_test"]["pooled"]["confusion_matrix"]:
        raise ValueError("Gallery cells, cases or test confusion disagree with study")
    manifest = {"status": "complete_diagnostic_not_biologically_validated",
                "cases": 9, "channels": list(GIGATIME_CHANNELS),
                "classes": list(CLASSES), "records": records,
                "new_test_balanced_accuracy": study["new_test"]["pooled"]["balanced_accuracy"],
                "new_test_matched_cells": study["new_test"]["pooled"]["n"],
                "new_test_confusion_matrix": total_confusion.tolist(),
                "publication_limit": "Exploratory diagnosis only. Technical DAPI/shadow passes do not validate virtual marker identity; source polygons can enclose multiple cells; accuracy conditions on conservative StarDist/expert matching. Development and fit images were used during model development."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sample_ids = [row["sample_id"] for row in records]
    for start in (0, 8, 16):
        files = [f"{sample}--marker-{index:02d}-contrast.webp"
                 for index in range(start, min(start + 8, 23)) for sample in sample_ids]
        contact(asset, files, args.output / f"contact-markers-{start:02d}.webp",
                (len(files) + 8) // 9, 9, label="marker")
    files = [f"{sample}--{name}.webp" for sample in sample_ids
             for name in ("he", "source-expert", "detected", "matched-expert",
                          "predicted", "errors")]
    contact(asset, files, args.output / "contact-classifier.webp", 9, 6,
            label="panel")
    (args.output / "VIEWER_COMPLETE.json").write_text(json.dumps({
        "status": "complete", "cases": 9, "channels": 23,
        "test_confusion_verified": True, "all_469_predictions_verified": True,
        "requires_manual_review_before_publication": True}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quantification", "prepared", "inference", "fit-detected",
                 "eval-detected", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = build(parser.parse_args())
    print(json.dumps({"cases": result["cases"],
                      "new_test_balanced_accuracy":
                          result["new_test_balanced_accuracy"]}))
