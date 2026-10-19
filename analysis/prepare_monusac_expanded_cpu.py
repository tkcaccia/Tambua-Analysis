#!/usr/bin/env python3
"""CPU-stage only the new 256-pixel MoNuSAC expansion candidates.

No GigaTIME inference or classifier fitting occurs here. Source RGB is decoded
from scanner SVS and checked against the companion TIFF by Tambua's preparer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-",
                                     delete=False) as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        staged = Path(stream.name)
    staged.replace(path)


def stage(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.tambua_scripts))
    from prepare_monusac_transfer import prepare_one

    candidate = json.loads(args.candidate.read_text())
    current = json.loads(args.current_cohort.read_text())
    audit_paths = {"train": args.train_audit, "test": args.test_audit}
    if any(sha(audit_paths[split]) != candidate["source_sha256"][f"{split}_audit"]
           for split in ("train", "test")) or \
            sha(args.current_cohort) != candidate["source_sha256"]["current_120_cohort"]:
        raise ValueError("Input audits or current cohort changed after candidate freeze")
    known = {r["sample_id"] for r in current["records"]}
    selected = [r for r in candidate["records"] if r["sample_id"] not in known]
    if len(selected) != 93 or sum(r["split"] == "train" for r in selected) != 59 or \
            sum(r["split"] == "test" for r in selected) != 34:
        raise ValueError("Expansion case count or split changed")
    audited = {split: {r["sample_id"]: r for r in
                       json.loads(path.read_text())["records"]}
               for split, path in audit_paths.items()}
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(args.output)
    samples = args.output / "samples"
    samples.mkdir(parents=True)
    records = []
    with ZipFile(args.train_zip) as train_zip, ZipFile(args.test_zip) as test_zip:
        archives = {"train": train_zip, "test": test_zip}
        for case in selected:
            split, sample = case["split"], case["sample_id"]
            original = dict(audited[split][sample], split=split)
            if any(original[key] != case[key] for key in
                   ("sample_id", "patient_id", "organ", "width", "height", "mpp")):
                raise ValueError(f"Candidate and source audit disagree: {sample}")
            folder = samples / sample
            if folder.exists():
                required = ("he.ome.tif", "expert_nuclei.ome.tif",
                            "expert_anchored_cell_proxy.ome.tif", "expert_types.csv",
                            "provenance.json")
                if any(not (folder / name).is_file() for name in required):
                    raise ValueError(f"Incomplete CPU checkpoint: {sample}")
                row = json.loads((folder / "provenance.json").read_text())
                if any(row[key] != case[key] for key in
                       ("sample_id", "patient_id", "organ", "split", "width", "height", "mpp")):
                    raise ValueError(f"CPU checkpoint identity mismatch: {sample}")
            else:
                row = prepare_one(archives[split], original, folder)
            if row["source_color_reference"] != "scanner_svs_rgb":
                raise ValueError(f"Scanner RGB not verified: {sample}")
            records.append(row)
            print(f"Prepared {len(records)}/{len(selected)} {split} {sample}", flush=True)
    marker = {"status": "cpu_prepared_only_no_gigatime_or_qc",
              "candidate_sha256": sha(args.candidate), "cases": len(records),
              "train": sum(r["split"] == "train" for r in records),
              "test": sum(r["split"] == "test" for r in records),
              "all_scanner_rgb": all(r["source_color_reference"] == "scanner_svs_rgb"
                                     for r in records),
              "records": records}
    atomic_json(args.output / "PREP_COMPLETE.json", marker)
    return {key: marker[key] for key in ("status", "cases", "train", "test",
                                          "all_scanner_rgb")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "current-cohort", "train-audit", "test-audit",
                 "train-zip", "test-zip", "tambua-scripts", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="Verify and reuse complete CPU checkpoints")
    print(json.dumps(stage(ap.parse_args())))
