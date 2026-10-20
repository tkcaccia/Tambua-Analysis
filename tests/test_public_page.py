"""Keep the public page limited to currently interpretable image evidence."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.values.extend(value for key, value in attrs if key in {"src", "href"} and value)


def test_page_labels_corrected_benchmark_and_quality_limits() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert "corrected-monusac.html" in html
    assert "detected-cells.html" in html
    assert "47 / 120" in html
    assert "94.26%" in html
    assert "both matched macrophages were missed" in html
    assert "no matched neutrophils" in html
    assert "four-class balanced accuracy" in html.lower()
    assert 'id="dapi-qc"' not in html
    for stale in ("85.2%", "pre-fix", "monusac-matched/", "monusac-transfer/", "orion-crc33-01/"):
        assert stale not in html.lower()


def test_local_page_assets_exist() -> None:
    refs = References()
    refs.feed(PAGE.read_text(encoding="utf-8"))
    assert refs.values
    for ref in refs.values:
        if ref.startswith("#") or urlparse(ref).scheme:
            continue
        assert (PAGE.parent / ref).is_file(), ref


def test_corrected_gallery_assets_cover_every_test_image() -> None:
    import json

    gallery = PAGE.parent / "corrected-monusac.html"
    assert gallery.is_file()
    revised = json.loads((PAGE.parent / "assets/monusac-corrected/stardist-nuclear-qc-summary.json").read_text())
    assert len(revised["records"]) == 120
    assert sum(all(value == "pass" for value in row["stardist_nuclear_qc"].values())
               for row in revised["records"]) == 47
    assert "stardist-nuclear-qc-summary.json" in gallery.read_text()
    for split, folder, expected in (("train", "monusac-corrected-train", 88),
                                    ("test", "monusac-corrected", 32)):
        assets = PAGE.parent / "assets" / folder
        data = json.loads((assets / "results.json").read_text())
        assert len(data["cases"]) == expected
        assert all(case["split"] == split for case in data["cases"])
        assert sum(case["native_nuclear_qc"]["status"] == "pass"
                   for case in data["cases"]) == (40 if split == "train" else 10)
        assert sum(case["accepted_qc"] for case in data["cases"]) == (38 if split == "train" else 10)
        for case in data["cases"]:
            assert case["source_color_reference"] == "scanner_svs_rgb"
            for panel in ("he", "expert", "predicted", "errors", "dapi", "ck", "cd3", "cd68"):
                assert (assets / f'{case["sample_id"]}-{panel}.webp').is_file()

    strict = json.loads((PAGE.parent / "assets/monusac-corrected/qc-strict-report.json").read_text())
    assert strict["qc_selection"]["train_pass"] == 38
    assert strict["qc_selection"]["test_pass"] == 10
    test_data = json.loads((PAGE.parent / "assets/monusac-corrected/results.json").read_text())
    assert test_data["strict_qc_test"]["confusion_matrix"] == strict["test"]["confusion_matrix"]


def test_detected_cell_gallery_has_complete_predictions_and_qualified_score() -> None:
    import json
    root = PAGE.parent / "assets/monusac-detected"
    gallery = PAGE.parent / "detected-cells.html"
    html = gallery.read_text(encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "study-summary.json").read_text())
    assert manifest["cases"] == len(manifest["records"]) == 47
    assert report["fit_images"] == 33
    assert report["development_images_not_used"] == 4
    assert report["organizer_test_images"] == 10
    assert report["matched_test_cells"] == 3975
    assert report["all_detected_test_cells_predicted"] == 5916
    metrics = report["test_metrics_on_conservatively_matched_cells"]
    assert metrics["balanced_accuracy"] is None
    assert metrics["support"]["neutrophil"] == 0
    assert metrics["recall"]["macrophage"] == 0
    assert "Rare-cell validation remains incomplete" in html
    assert sum(row["detected_cells_predicted"] for row in manifest["records"]
               if row["split"] == "test") == 5916
    for row in manifest["records"]:
        for panel in ("detected", "predicted", "matched-errors"):
            path = root / f'{row["sample_id"]}-{panel}.webp'
            assert path.stat().st_size > 1000
            assert path.read_bytes()[:4] == b"RIFF"

    markers = PAGE.parent / "assets/monusac-all-markers"
    marker_manifest = json.loads((markers / "manifest.json").read_text())
    contacts = json.loads((markers / "CONTACTS_COMPLETE.json").read_text())
    assert marker_manifest["cases"] == 47
    assert len(marker_manifest["channel_names"]) == 23
    assert contacts["cases_per_sheet"] == 47
    assert len(contacts["channels"]) == 23
    assert "Open the 47-image overview" in html
    assert {row["sample_id"] for row in marker_manifest["records"]} == \
           {row["sample_id"] for row in manifest["records"]}
    assert "All 23 spatial GigaTIME channels" in html
    for row in marker_manifest["records"]:
        assert row["all_channel_seam_qc"] == "pass"
        assert row["marker_distribution_qc"] in {"pass", "review"}
        for index in range(23):
            path = markers / f'{row["sample_id"]}--{index:02d}.webp'
            assert path.stat().st_size > 1000
            assert path.read_bytes()[:4] == b"RIFF"
    for index in range(23):
        contact = markers / f"contact-{index:02d}.webp"
        assert contact.stat().st_size > 1000
        assert contact.read_bytes()[:4] == b"RIFF"


def test_prepared_expansion_reports_rare_classes_without_claiming_results() -> None:
    import json

    audit = json.loads((PAGE.parent / "assets/monusac-corrected/expansion-rare-coverage.json").read_text())
    assert audit["additional_images"] == 93
    assert audit["counts"]["train_additional"]["images"] == 59
    assert audit["counts"]["test_additional"]["images"] == 34
    assert audit["counts"]["train_additional"]["source_polygons_by_class"]["neutrophil"] == 201
    assert audit["counts"]["test_additional"]["source_polygons_by_class"]["neutrophil"] == 46
    assert "precede independent nuclear detection" in audit["limit"]
    page = PAGE.read_text(encoding="utf-8")
    assert "source-region counts do not reflect the nine QC-passing images" in page
    assert "pre-GigaTIME, pre-QC candidates" in page


def test_expanded_exploratory_gallery_covers_every_qc_passing_field_and_marker() -> None:
    import json

    page = (PAGE.parent / "expanded-monusac-classifier.html").read_text(encoding="utf-8")
    root = PAGE.parent / "assets/monusac-expanded-exploratory"
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "study-summary.json").read_text())
    assert manifest["status"] == "complete_diagnostic_not_biologically_validated"
    assert report["publication_status"] == \
        "exploratory_do_not_publish_as_validated_cell_type_accuracy"
    assert len(manifest["records"]) == 9
    assert len(manifest["channels"]) == 23
    assert {row["split"] for row in manifest["records"]} == \
        {"train", "development", "test"}
    assert sum(row["detected_cells"] for row in manifest["records"]) == 469
    assert manifest["new_test_matched_cells"] == 184
    assert manifest["new_test_confusion_matrix"] == \
        report["new_test"]["pooled"]["confusion_matrix"]
    assert "44.6%" in page and "0% recall for macrophages and neutrophils" in page
    for row in manifest["records"]:
        sample = row["sample_id"]
        for name in ("he", "source-expert", "detected", "matched-expert",
                     "predicted", "errors"):
            assert (root / "assets" / f"{sample}--{name}.webp").stat().st_size > 100
        for index in range(23):
            for mode in ("linear", "contrast"):
                assert (root / "assets" /
                        f"{sample}--marker-{index:02d}-{mode}.webp").stat().st_size > 100


def test_expansion_input_preflight_covers_every_prepared_image_without_claiming_qc() -> None:
    import json

    page = (PAGE.parent / "expansion-input-preflight.html").read_text(encoding="utf-8")
    root = PAGE.parent / "assets/monusac-expansion-preflight"
    manifest = json.loads((root / "manifest.json").read_text())
    ids = [sample for part in manifest["pages"] for sample in part["sample_ids"]]
    assert manifest["cases"] == len(ids) == len(set(ids)) == 93
    assert len(manifest["pages"]) == 4
    assert "Input preflight only" in page and "not virtual markers" in page
    for part in manifest["pages"]:
        path = root / part["file"]
        assert path.stat().st_size > 10000
        assert path.read_bytes()[:4] == b"RIFF"
        assert part["file"] in page

    geometry = json.loads((PAGE.parent / "assets/monusac-corrected/expansion-geometry-audit.json").read_text())
    assert geometry["images"] == 93
    assert geometry["by_class"]["macrophage"]["count"] == 414
    assert len(geometry["large_source_macrophage_regions"]) == 70
    assert "manual-review flag" in geometry["threshold_note"]


def test_rare_balance_audit_is_fit_patient_held_out_development_only() -> None:
    import json

    audit = json.loads((PAGE.parent / "assets/monusac-development-balance/RARE_BALANCE_AUDIT.json").read_text())
    assert audit["device"] == "CPU"
    assert len(audit["patients"]) == 21
    assert "fit cells only" in audit["scope"]
    assert audit["strategies"]["original"]["support"]["neutrophil"] == 8
    assert audit["strategies"]["original"]["recall"]["neutrophil"] == 0
    assert all(item["balanced_accuracy"] < .9 for item in audit["strategies"].values())
    assert "cannot establish final four-class accuracy" in audit["limit"]


def test_development_qc_fail_audit_does_not_promote_rejected_orion_cases() -> None:
    import json

    audit = json.loads((PAGE.parent / "assets/orion-pixel-dapi/development-qc-fail-audit.json").read_text())
    assert audit["patients"] == 4
    assert audit["device"] == "CPU"
    assert {row["patient"] for row in audit["records"]} == {"CRC01", "CRC03", "CRC15", "CRC16"}
    assert all(row["native_qc_status"] == "fail" and not row["original_case_technical_qc_passed"]
               for row in audit["records"])
    assert "not independent validation" in audit["limit"]
    assert "remain excluded" in PAGE.read_text(encoding="utf-8")


def test_orion_pixel_gallery_is_registered_and_complete() -> None:
    import json

    page = PAGE.parent / "orion-pixel-dapi.html"
    assert page.is_file()
    root = PAGE.parent / "assets/orion-pixel-dapi"
    manifest = json.loads((root / "manifest.json").read_text())
    scores = json.loads((root / "summary.json").read_text())
    assert scores["patients"] == len(manifest["records"]) == 10
    assert len({row["patient"] for row in manifest["records"]}) == 10
    assert manifest["scores"]["equal_patient_median_spearman"] == scores["equal_patient_median_spearman"]
    for row in manifest["records"]:
        assert row["source_mpp"] == 0.325
        assert row["virtual_mpp"] == 0.25
        assert 0 < row["pixel_spearman_rho"] < 1
        for panel in ("he", "measured-Hoechst", "virtual-DAPI", "pixel-overlap"):
            image = root / f'{row["patient"]}-{panel}.webp'
            assert image.stat().st_size > 1000
            assert image.read_bytes()[:4] == b"RIFF"

    replay = json.loads((root / "spatial-compartment-dapi-replay.json").read_text())
    aggregate = {item["compartment"]: item for item in replay["equal_patient_aggregates"]}
    assert replay["patients"] == 10 and replay["markers"] == ["DAPI"]
    assert aggregate["full_field"]["equal_patient_median_spearman"] == scores["equal_patient_median_spearman"]
    assert aggregate["registered_nuclei"]["equal_patient_median_spearman"] < .25
    assert "0.174" in page.read_text(encoding="utf-8")


def test_orion_measured_protein_gallery_uses_real_image_pixels() -> None:
    import json

    page = (PAGE.parent / "orion-measured-preflight.html").read_text(encoding="utf-8")
    root = PAGE.parent / "assets/orion-measured-preflight"
    manifest = json.loads((root / "COHORT_COMPLETE.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["patients"] == 10 and manifest["measured_marker_images"] == 100
    assert len({row["patient"] for row in manifest["records"]}) == 10
    assert all(row["markers"] == 10 and row["source_mpp"] == .325
               for row in manifest["records"])
    assert "Measured-source inspection only" in page
    assert "not GigaTIME predictions" in page
    for row in manifest["records"]:
        path = root / f'{row["patient"]}.webp'
        assert path.stat().st_size > 10000
        assert path.read_bytes()[:4] == b"RIFF"
