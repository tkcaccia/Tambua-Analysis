#!/usr/bin/env python3
"""Describe frozen GigaTIME nuclear-QC failures without changing any gate.

This audit is valid on a partial CPU bundle. Its counts are diagnostic, not a
replacement for image-by-image visual review or a final classifier cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(root: Path) -> dict:
    records = []
    identities = []
    for case_path in sorted(root.glob("*/CASE_COMPLETE.json")):
        qc_path = case_path.with_name("stardist_qc.json")
        if not qc_path.is_file():
            raise FileNotFoundError(qc_path)
        case = json.loads(case_path.read_text())
        qc = json.loads(qc_path.read_text())
        if case["sample_id"] != case_path.parent.name or case["nuclear_qc"]["status"] != qc["status"]:
            raise ValueError(f"Mismatched QC for {case_path.parent.name}")
        identities.append(f"{case_path.parent.name} {hashlib.sha256(case_path.read_bytes()).hexdigest()} "
                          f"{hashlib.sha256(qc_path.read_bytes()).hexdigest()}")
        records.append((case, qc))

    gates = ("alignment_status", "border_shadow_status", "spatial_shadow_status")
    report = {
        "status": "partial_diagnostic",
        "completed_cases": len(records),
        "source_identity_sha256": hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest(),
        "overall": dict(Counter(qc["status"] for _, qc in records)),
        "gate_counts": {gate: dict(Counter(qc[gate] for _, qc in records)) for gate in gates},
        "failed_alignment": sum(qc["status"] == "fail" and qc["alignment_status"] == "fail"
                                for _, qc in records),
        "failed_with_alignment_pass": sum(qc["status"] == "fail" and qc["alignment_status"] == "pass"
                                          for _, qc in records),
        "by_organ": {organ: {"cases": len(group),
                             "pass": sum(qc["status"] == "pass" for _, qc in group)}
                     for organ in sorted({case["organ"] for case, _ in records})
                     for group in [[(case, qc) for case, qc in records if case["organ"] == organ]]},
        "interpretation": ("Most failures may be nuclear alignment rather than the explicit border or spatial "
                           "shadow gates. These gate counts overlap and cannot be added. A visual near-pass "
                           "does not authorize threshold relaxation or establish protein specificity."),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end="")
