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
    assert "Native DAPI QC passed 50 images and failed 70" in html
    assert "conditional on expert-drawn regions and QC selection" in html
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
    data = json.loads((PAGE.parent / "assets/monusac-corrected/results.json").read_text())
    assert len(data["cases"]) == 32
    assets = PAGE.parent / "assets/monusac-corrected"
    for case in data["cases"]:
        assert case["source_color_reference"] == "scanner_svs_rgb"
        for panel in ("he", "expert", "predicted", "errors", "dapi", "ck", "cd3", "cd68"):
            assert (assets / f'{case["sample_id"]}-{panel}.webp').is_file()
