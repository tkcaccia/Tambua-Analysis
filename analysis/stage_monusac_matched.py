"""Stage one immutable, patient-disjoint MoNuSAC GigaTIME bundle on Chiamaka.

This is CPU-only. It reuses prior completed outputs only after bytewise input
identity checks; the GPU launcher processes only the remaining images.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from xml.etree import ElementTree as ET

import numpy as np
import tifffile


OLD = Path("/mnt/sata_ssd/Tambua-monusac-transfer-2026-10-19-v1")
OLD_PREP = Path("/mnt/sata_ssd/Tambua-monusac-transfer-2026-10-19-prep")
PREP = Path("/mnt/sata_ssd/Tambua-monusac-matched-2026-10-19-prep-v2")
CASE = Path("/mnt/sata_ssd/Tambua-monusac-matched-2026-10-19-v1")
INPUTS = ("he.ome.tif", "expert_nuclei.ome.tif",
          "expert_anchored_cell_proxy.ome.tif", "expert_types.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not (OLD / "evidence/BUNDLE_COMPLETE.json").is_file():
        raise ValueError("Earlier 37-image GPU bundle is not complete")
    cohort = json.loads((PREP / "COHORT.json").read_text())
    old_cohort = json.loads((OLD / "input/COHORT.json").read_text())
    rows = cohort["records"]
    old_ids = {row["sample_id"] for row in old_cohort["records"]}
    new_ids = [row["sample_id"] for row in rows if row["sample_id"] not in old_ids]
    if len(rows) != 120 or len(old_ids) != 37 or len(new_ids) != 83:
        raise ValueError("Cohort cardinality changed")
    for row in rows:
        sample = row["sample_id"]
        if sample not in old_ids:
            continue
        if not (OLD / "output" / sample / "RUN_COMPLETE.json").is_file():
            raise ValueError(f"Prior output lacks completion marker: {sample}")
        if not (OLD / "output" / sample / "VALIDATION.json").is_file():
            raise ValueError(f"Prior output lacks validation: {sample}")
        old_provenance = json.loads((OLD_PREP / sample / "provenance.json").read_text())
        new_provenance = json.loads((PREP / sample / "provenance.json").read_text())
        if old_provenance["source_tiff_sha256"] != new_provenance["source_image_sha256"]:
            raise ValueError(f"Source image differs for reused {sample}")
        for name in INPUTS:
            before, after = OLD_PREP / sample / name, PREP / sample / name
            if name.endswith(".csv"):
                equal = sha256(before) == sha256(after)
            else:
                with tifffile.TiffFile(before) as a, tifffile.TiffFile(after) as b:
                    ap = ET.fromstring(a.ome_metadata).find(".//{*}Pixels")
                    bp = ET.fromstring(b.ome_metadata).find(".//{*}Pixels")
                    equal = (np.array_equal(a.asarray(), b.asarray()) and
                             ap.attrib["PhysicalSizeX"] == bp.attrib["PhysicalSizeX"] and
                             ap.attrib["PhysicalSizeY"] == bp.attrib["PhysicalSizeY"])
            if not equal:
                raise ValueError(f"Prepared input differs for reused {sample}: {name}")
    CASE.mkdir(parents=True, exist_ok=False)
    for subdir in ("input", "code", "output", "evidence"):
        (CASE / subdir).mkdir()
    shutil.copy2(PREP / "COHORT.json", CASE / "input/COHORT.json")
    (CASE / "input/NEW_SAMPLES.txt").write_text("\n".join(new_ids) + "\n")
    (CASE / "prepared").symlink_to(PREP)
    for sample in old_ids:
        (CASE / "output" / sample).symlink_to(OLD / "output" / sample)
    for name in ("run_gigatime_on_crop.py", "gigatime_hardware.py",
                 "gigatime_resolution.py", "gigatime_seam_qc.py",
                 "model_provenance.py", "validate_sample.py"):
        shutil.copy2(OLD / "code" / name, CASE / "code" / name)
    report = {"status": "staged", "total_images": 120, "reused_images": 37,
              "new_gpu_images": 83,
              "cohort_sha256": sha256(CASE / "input/COHORT.json"),
              "prior_bundle": str(OLD), "new_samples_sha256": sha256(CASE / "input/NEW_SAMPLES.txt")}
    (CASE / "evidence/STAGING.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
