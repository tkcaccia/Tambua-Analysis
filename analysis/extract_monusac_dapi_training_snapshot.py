#!/usr/bin/env python3
"""Extract completed training DAPI and independent StarDist masks, CPU only.

The live inference bundle is read-only. No test images or labels are exported.
Each source is a checkpointed, identity-matched case on the fixed input cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile
import zarr


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def run(prepared: Path, inference: Path, output: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Training snapshot must hide GPUs")
    if output.exists():
        raise FileExistsError(output)
    plan_path = prepared / "PREP_COMPLETE.json"
    plan = json.loads(plan_path.read_text())
    identity = json.loads((inference / "BUNDLE_IDENTITY.json").read_text())
    if plan["cases"] != 93 or not plan["all_scanner_rgb"] or \
            identity["device"] != "CPU" or \
            identity["prepared_sha256"] != sha(plan_path) or \
            set(identity["case_order"]) != {r["sample_id"] for r in plan["records"]}:
        raise ValueError("Frozen expansion identity changed")
    cohort = {r["sample_id"]: r for r in plan["records"]}
    complete = [name for name in identity["case_order"]
                if (inference / name / "CASE_COMPLETE.json").is_file()]
    selected = [name for name in complete if cohort[name]["split"] == "train"]
    if not selected:
        raise ValueError("No checkpointed training cases")
    output.mkdir(parents=True)
    records = []
    for name in selected:
        case = inference / name
        done = json.loads((case / "CASE_COMPLETE.json").read_text())
        qc = json.loads((case / "stardist_qc.json").read_text())
        store = zarr.open(str(case / "gigatime_output" / "gigatime_probs.zarr"), mode="r")
        data = store["0"]
        meta = store.attrs["gigatime"]
        if done["sample_id"] != name or qc["status"] != done["nuclear_qc"]["status"] or \
                data.dtype != np.uint16 or data.shape[0] != 23 or \
                abs(float(meta["effective_mpp"]) - .25) > 1e-6 or \
                tuple(done["image_shape_yx"]) != tuple(data.shape[1:]):
            raise ValueError(f"Saved training DAPI identity changed: {name}")
        source = prepared / "samples" / name / "he.ome.tif"
        if sha(source) != done["source_rgb_sha256"]:
            raise ValueError(f"Scanner RGB source hash changed: {name}")
        mask_path = Path(qc["nuclei_mask"])
        mask = tifffile.imread(mask_path)
        dapi = np.asarray(data[0])
        if mask.shape != tuple(meta["original_shape_yx"]):
            raise ValueError(f"Independent nuclei/source grid changed: {name}")
        np.savez_compressed(output / f"{name}.npz", dapi=dapi,
                            stardist_source_mask=mask)
        records.append({"sample_id": name, "patient_id": cohort[name]["patient_id"],
                        "organ": cohort[name]["organ"], "split": "train",
                        "source_mpp": cohort[name]["mpp"],
                        "source_rgb_sha256": done["source_rgb_sha256"],
                        "source_mask_sha256": sha(mask_path),
                        "source_nuclear_qc_status": qc["status"],
                        "source_nuclear_auc": qc["auc"],
                        "array_sha256": sha(output / f"{name}.npz")})
    report = {"status":"training_only_read_only_snapshot",
              "device":"CPU", "completed_cases_at_snapshot":len(complete),
              "training_cases":len(records), "prepared_sha256":sha(plan_path),
              "records":records,
              "limit":"Only completed training fields. Any correction selected here must be frozen before inspecting test fields; the original virtual markers remain unchanged."}
    (output/"SNAPSHOT.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--inference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=run(a.prepared,a.inference,a.output)
    print(json.dumps({"cases":r["training_cases"],"snapshot":r["completed_cases_at_snapshot"]}))
