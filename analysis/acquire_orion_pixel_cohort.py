#!/usr/bin/env python3
"""Acquire measured ORION Hoechst pixels for every reserved QC-passing patient.

CPU-only and fail-fast. Each patient writes a source manifest only after its
window completes; a final marker is written only when every patient completed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from acquire_orion_pixel_if import acquire


def atomic_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-",
                                     delete=False) as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        staged = Path(stream.name)
    staged.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--gallery-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    gallery = json.loads(args.gallery_manifest.read_text())
    patients = sorted({row["patient"] for row in gallery["records"]})
    if len(patients) != 10:
        raise ValueError("Expected ten reserved ORION patients with complete QC")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for patient in patients:
        prepared = args.study / "cpu-prep" / patient
        target = args.output / patient
        result = acquire(prepared, target, ["Hoechst"])
        records.append({"patient": patient, "source": result["source_url"],
                        "sha256": result["channels"][0]["sha256"]})
        print(f"Measured Hoechst acquired: {patient}", flush=True)
    atomic_json(args.output / "COHORT_COMPLETE.json", {
        "scope": "Ten reserved ORION patients with pre-existing virtual-DAPI and seam QC passes",
        "measured": "Registered raw pixel-level Hoechst, not per-cell painted values",
        "patients": records,
    })


if __name__ == "__main__":
    main()
