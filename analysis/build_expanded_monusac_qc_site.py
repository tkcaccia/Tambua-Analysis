#!/usr/bin/env python3
"""Publish only reviewed, identity-checked MoNuSAC expansion QC previews.

This is a diagnostic gallery. Its image rows are derived from the frozen
five-panel contact sheets, not from cell-type predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


HEADER = 54
ROW = 304
WIDTH = 1250


def reviewed_ids(evidence: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    paths = sorted(evidence.glob("monusac-expanded-output*/**/*visual-review*.json"))
    for path in paths:
        report = json.loads(path.read_text())
        ids = []
        for key in ("reviewed_cases", "records"):
            for item in report.get(key, []):
                ids.append(item.get("sample_id") if isinstance(item, dict) else item)
        if report.get("sample_id"):
            ids.append(report["sample_id"])
        for sample in ids:
            if not isinstance(sample, str) or sample in out:
                raise ValueError(f"Missing or duplicate manual review: {sample}")
            out[sample] = str(path)
    return out


def build(site_assets: Path, evidence: Path) -> dict:
    source = site_assets / "manifest.json"
    data = json.loads(source.read_text())
    records = data["records"]
    if data["status"] != "complete" or len(records) != 93 or \
            data["qc_status_counts"] != {"fail": 84, "pass": 9}:
        raise ValueError("Frozen 93-image inference or QC summary changed")
    reviews = reviewed_ids(evidence)
    samples = {row["sample_id"] for row in records}
    if set(reviews) != samples or len(samples) != 93:
        raise ValueError(f"Manual review coverage differs: {sorted(samples-set(reviews))}")
    by_id = {row["sample_id"]: row for row in records}
    case_dir = site_assets / "cases"
    if case_dir.exists() or (site_assets / "site-manifest.json").exists():
        raise FileExistsError("Diagnostic site output exists; preserve published evidence")
    case_dir.mkdir()
    pages = []
    for page in data["pages"]:
        image = site_assets / page["file"]
        with Image.open(image) as sheet:
            if sheet.width != WIDTH or sheet.height != HEADER + len(page["sample_ids"]) * ROW:
                raise ValueError(f"Contact-sheet geometry changed: {image}")
            for index, sample in enumerate(page["sample_ids"]):
                if sample not in by_id:
                    raise ValueError(f"Unknown case on sheet: {sample}")
                y = HEADER + index * ROW
                crop = sheet.crop((0, y, WIDTH, y + ROW))
                crop.save(case_dir / f"{sample}.webp", "WEBP", quality=90)
                pages.append({"sample_id": sample, "organ": by_id[sample]["organ"],
                              "split": by_id[sample]["split"],
                              "qc_status": by_id[sample]["qc_status"],
                              "auc": by_id[sample]["dapi_auc"],
                              "nuclear_boundary_contrast": by_id[sample]["dapi_boundary_contrast"],
                              "source_mpp_um": by_id[sample]["source_mpp"],
                              "model_mpp_um": by_id[sample]["model_mpp"],
                              "image": f"cases/{sample}.webp",
                              "manual_review": reviews[sample]})
    if {row["sample_id"] for row in pages} != samples or len(pages) != 93:
        raise ValueError("Contact sheets omit or duplicate a case")
    report = {"status": "diagnostic_gallery_complete",
              "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "images": 93, "qc_pass": 9, "qc_fail": 84,
              "all_images_manually_reviewed_at_contact_sheet_scale": True,
              "scope": "Scanner RGB, virtual DAPI, StarDist overlay, virtual CK and CD68 for every additional annotated MoNuSAC image",
              "limit": "Contact-sheet review is not full-resolution marker-specific validation. QC failures are excluded from fastPLS training and test claims; technical passes remain subject to 23-channel review.",
              "cases": pages}
    (site_assets / "site-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site-assets", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    a = ap.parse_args()
    result = build(a.site_assets, a.evidence)
    print(json.dumps({key: result[key] for key in ("status", "images", "qc_pass", "qc_fail")}))
