import json

import pytest

from analysis.score_monusac_matched import case_qc_status, require_scanner_rgb


def test_archived_scores_need_native_nuclear_qc_before_reuse(tmp_path):
    assert case_qc_status(tmp_path, "case") == "missing"
    output = tmp_path / "gigatime_output"
    quant = tmp_path / "quantification"
    output.mkdir()
    quant.mkdir()
    report = output / "output_gigatime_nuclear_qc.json"
    summary = quant / "output_gigatime_integrated_quantification_summary.json"
    report.write_text(json.dumps({"status": "fail"}))
    summary.write_text(json.dumps({"technical_qc_passed": False}))
    assert case_qc_status(tmp_path, "case") == "fail"
    report.write_text(json.dumps({"status": "pass"}))
    summary.write_text(json.dumps({"technical_qc_passed": True}))
    assert case_qc_status(tmp_path, "case") == "pass"


def test_new_scores_reject_misordered_companion_tiff_cohort():
    rows = [{"sample_id": "old", "source_image_member": "old.tif"}]
    with pytest.raises(ValueError, match="scanner-SVS-corrected RGB"):
        require_scanner_rgb(rows, False)
    require_scanner_rgb(rows, True)
    require_scanner_rgb([{"source_color_reference": "scanner_svs_rgb"}], False)
