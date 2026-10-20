#!/usr/bin/env python3
"""Verify the ten registered ORION measured-marker image sets before comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile

from render_orion_measured_preflight import MARKERS


def verify(study: Path, original_dapi: Path, measured: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Measured-image verification must explicitly hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    original = json.loads((original_dapi / "COHORT_COMPLETE.json").read_text())
    patients = [item["patient"] for item in original["patients"]]
    if len(patients) != 10 or len(set(patients)) != 10:
        raise ValueError("Expected ten distinct original QC-pass patients")
    if set(item.name for item in measured.iterdir() if item.is_dir() and item.name != "previews") != set(patients):
        raise ValueError("Measured image folders do not match the registered cohort")
    records = []
    for patient in patients:
        prep = json.loads((study / "cpu-prep" / patient / "PROVENANCE.json").read_text())
        source = json.loads((measured / patient / "MEASURED_PIXEL_SOURCE.json").read_text())
        old = json.loads((original_dapi / patient / "MEASURED_PIXEL_SOURCE.json").read_text())
        if source["patient"] != patient or any(source[k] != prep[k]
                                                for k in ("x", "y", "size_px", "mpp_x", "mpp_y")):
            raise ValueError(f"Registered window or scale changed: {patient}")
        if source["source_url"] != old["source_url"] or \
                source["channels"][0]["sha256"] != old["channels"][0]["sha256"]:
            raise ValueError(f"Hoechst source or raw pixels changed: {patient}")
        if tuple(item["marker"] for item in source["channels"]) != MARKERS:
            raise ValueError(f"Measured marker panel changed: {patient}")
        for channel in source["channels"]:
            path = measured / patient / f'{channel["marker"]}.tif'
            pixels = tifffile.imread(path)
            if pixels.shape != (prep["size_px"], prep["size_px"]) or pixels.dtype != np.uint16 or \
                    hashlib.sha256(pixels.tobytes()).hexdigest() != channel["sha256"]:
                raise ValueError(f"Measured image geometry or hash changed: {patient}/{channel['marker']}")
        records.append({"patient": patient, "source_url": source["source_url"],
                        "source_mpp": source["mpp_x"],
                        "window_xy": [source["x"], source["y"]],
                        "markers": len(source["channels"])})
    result = {"status": "complete", "device": "CPU", "patients": len(records),
              "measured_marker_images": sum(row["markers"] for row in records),
              "markers": MARKERS, "records": records,
              "scope": "Registered raw measured fluorescence pixels, not virtual marker predictions",
              "limit": "These measured images do not by themselves validate GigaTIME or cell typing."}
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(str(output) + ".tmp")
    staged.write_text(json.dumps(result, indent=2) + "\n")
    staged.replace(output)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, required=True)
    ap.add_argument("--original-dapi", type=Path, required=True)
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps({key: value for key, value in verify(args.study, args.original_dapi,
                                                           args.measured, args.output).items()
                      if key in ("status", "patients", "measured_marker_images")}))
