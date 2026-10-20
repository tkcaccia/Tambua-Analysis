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
