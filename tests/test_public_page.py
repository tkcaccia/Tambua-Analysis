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


def test_page_has_no_invalidated_marker_gallery_or_accuracy() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert "No accepted post-correction test score" in html
    assert "Quality decision: fail" in html
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
