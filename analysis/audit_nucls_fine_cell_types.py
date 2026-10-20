#!/usr/bin/env python3
"""Audit NuCLS corrected polylines and freeze a patient-separated cell cohort.

This CPU-only source audit does not run GigaTIME or fit a classifier. It uses
fine `main_classification` labels, which the convenient superclass masks hide.
Rectangular annotations are excluded because they are not nuclear boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq


SHARD_SHA256 = {
    "train-00000-of-00003.parquet": "9f59987fe978f8c8d9f1adf4fc87549d3d15de02a8241f4d403b46e2cfa52b46",
    "train-00001-of-00003.parquet": "abdfd25673abdfc03cbffd9b3d74eb79c5e08819baf9d259d90fc2b81261c3da",
    "train-00002-of-00003.parquet": "6bbc9f31954dc8f91f1d149dd3aa100a17596d61919cb2930f5c8e17e123a026",
}
TARGET = {
    "tumor_nonMitotic": "tumor_epithelial",
    "tumor_mitotic": "tumor_epithelial",
    "lymphocyte": "lymphocyte",
    "macrophage": "macrophage",
    "plasma_cell": "plasma_cell",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def annotation_counts(text: str) -> tuple[Counter, Counter, int, int]:
    counts: Counter[str] = Counter()
    raw: Counter[str] = Counter()
    rectangles = 0
    points = 0
    for row in csv.DictReader(io.StringIO(text)):
        if row["type"] == "rectangle":
            rectangles += 1
            continue
        if row["type"] == "point":
            points += 1
            continue
        if row["type"] != "polyline":
            raise ValueError(f"Unexpected NuCLS annotation type: {row['type']}")
        label = row["main_classification"]
        counts[TARGET.get(label, "excluded_" + label)] += 1
        raw[row["raw_classification"]] += 1
        xs = [int(x) for x in row["coords_x"].split(",")]
        ys = [int(y) for y in row["coords_y"].split(",")]
        if len(xs) != len(ys) or len(xs) < 3:
            raise ValueError("Polyline lacks a closed nuclear boundary")
    return counts, raw, rectangles, points


def informative_without_macrophage(row: dict) -> bool:
    counts = row["target_polyline_counts"]
    represented = sum(counts.get(name, 0) > 0 for name in
                      ("tumor_epithelial", "lymphocyte", "plasma_cell"))
    return counts.get("macrophage", 0) == 0 and represented >= 2 and \
        sum(counts.get(name, 0) for name in TARGET.values()) >= 5


def freeze(rows: list[dict], extra_train: int, extra_test: int) -> list[dict]:
    selected = []
    for split, extra in (("train", extra_train), ("test", extra_test)):
        subset = [r for r in rows if r["split"] == split]
        rare = [r for r in subset if r["target_polyline_counts"].get("macrophage", 0)]
        others = [r for r in subset if informative_without_macrophage(r)]
        others.sort(key=lambda r: hashlib.sha256(r["fov_id"].encode()).hexdigest())
        if len(others) < extra:
            raise ValueError(f"Too few informative non-macrophage {split} fields")
        for row in rare + others[:extra]:
            selected.append({**row, "selection": ("all_macrophage_polyline_fields" if
                                               row in rare else
                                               "deterministic_class_diverse_supplement")})
    return sorted(selected, key=lambda r: (r["split"], r["fov_id"]))


def run(source: Path, output: Path, extra_train: int = 50,
        extra_test: int = 20) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if extra_train < 0 or extra_test < 0:
        raise ValueError("Negative supplement size")
    rows = []
    for name, expected in SHARD_SHA256.items():
        path = source/name
        if sha(path) != expected:
            raise ValueError(f"NuCLS mirror shard hash changed: {name}")
        columns = ("fov_id", "tcga_patient", "fold_1", "height", "width",
                   "n_nuclei", "n_polyline", "n_rectangle", "annotations_csv")
        table = pq.read_table(path, columns=list(columns))
        for index, item in enumerate(table.to_pylist()):
            counts, raw, rectangles, points = annotation_counts(item["annotations_csv"])
            if sum(counts.values()) != item["n_polyline"] or \
                    rectangles != item["n_rectangle"] or \
                    sum(counts.values()) + rectangles + points != item["n_nuclei"]:
                raise ValueError(f"Mirrored metadata/count mismatch: {item['fov_id']}")
            rows.append({"fov_id": item["fov_id"],
                         "patient_id": item["tcga_patient"],
                         "split": item["fold_1"],
                         "source_shard": name, "source_row": index,
                         "height": item["height"], "width": item["width"],
                         "source_mpp_um": 0.2,
                         "target_polyline_counts": dict(counts),
                         "fine_raw_polyline_counts": dict(raw),
                         "excluded_rectangles": rectangles,
                         "excluded_points": points})
    ids = [r["fov_id"] for r in rows]
    train = {r["patient_id"] for r in rows if r["split"] == "train"}
    test = {r["patient_id"] for r in rows if r["split"] == "test"}
    if len(rows) != 1744 or len(set(ids)) != 1744 or len(train) != 102 or \
            len(test) != 22 or train & test or \
            {r["split"] for r in rows} != {"train", "test"}:
        raise ValueError("Expected corrected, patient-separated NuCLS fold 1")
    selected = freeze(rows, extra_train, extra_test)
    by_split = {}
    for split in ("train", "test"):
        group = [r for r in selected if r["split"] == split]
        counts = Counter()
        for row in group:
            counts.update({k: v for k, v in row["target_polyline_counts"].items()
                           if not k.startswith("excluded_")})
        by_split[split] = {"images": len(group),
                           "patients": len({r["patient_id"] for r in group}),
                           "macrophage_images": sum(r["target_polyline_counts"].get(
                               "macrophage", 0)>0 for r in group),
                           "target_polyline_counts": dict(counts)}
    report = {"status": "fixed_source_cohort_no_inference",
              "source": "NuCLS corrected single-rater, MedOtter Parquet mirror",
              "source_shard_sha256": SHARD_SHA256,
              "fov_total": len(rows), "train_patients_total": len(train),
              "test_patients_total": len(test),
              "selection": "All fields with an outlined macrophage, plus 50 train and 20 test non-macrophage fields with at least two represented target classes and five target polylines, sorted by SHA-256 of FOV ID. This is explicitly expert-label-enriched; not a prevalence-representative sample.",
              "annotation_rule": "Only CSV polyline boundaries. Main tumor_nonMitotic and tumor_mitotic map to tumor_epithelial; macrophage, lymphocyte and plasma_cell retain their fine labels. Other nuclei, rectangles and degenerate points are excluded from typed-cell targets.",
              "calibration": "NuCLS source coordinates are 0.2 µm/pixel; any future GigaTIME input must use physical resampling to its pinned 0.25 µm/pixel grid.",
              "by_split": by_split, "selected": selected,
              "limit": "A frozen source/annotation plan only. No GigaTIME output, StarDist QC, cell quantification, classifier prediction or score exists. NuCLS has no outlined test neutrophils and cannot validate that cell type."}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix+".tmp")
    tmp.write_text(json.dumps(report, indent=2)+"\n")
    tmp.replace(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra-train", type=int, default=50)
    p.add_argument("--extra-test", type=int, default=20)
    a = p.parse_args()
    x = run(a.source, a.output, a.extra_train, a.extra_test)
    print(json.dumps({"selected_images": len(x["selected"]),
                      "by_split": x["by_split"]}))
