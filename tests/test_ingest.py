"""
test_ingest.py
Tests for URL and PDF ingestion. URL tests mock requests.get directly
(no live network calls -- this sandbox can't reach arbitrary external
URLs anyway, and tests shouldn't depend on a live website's html
staying the same). PDF tests generate a real small PDF on the fly with
reportlab and round-trip it through extract_from_pdf for a genuine
end-to-end check of that path.
"""

import io
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ingest import extract_from_url, extract_from_pdf, IngestError


SAMPLE_HTML = """
<html>
<head><title>Test Article About Trains</title></head>
<body>
<nav>Home About Contact</nav>
<header>Site Header</header>
<article>
<h1>Test Article About Trains</h1>
<p>This is the first real paragraph of the article, long enough to pass the length filter.</p>
<p>This is the second real paragraph, also reasonably long and full of content.</p>
<script>console.log('tracking pixel junk');</script>
<li>A list item that is also long enough to count as real content here.</li>
</article>
<footer>Copyright 2026</footer>
</body>
</html>
"""


def _make_mock_response(html: str, status_code: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.raw.read = MagicMock(return_value=html.encode("utf-8"))
    return mock_resp


def test_extract_from_url_rejects_invalid_url():
    with pytest.raises(IngestError, match="valid URL"):
        extract_from_url("not-a-url")


@patch("ingest.requests.get")
def test_extract_from_url_strips_noise_tags_and_keeps_content(mock_get):
    mock_get.return_value = _make_mock_response(SAMPLE_HTML)
    doc = extract_from_url("https://example.com/article")

    assert doc.title == "Test Article About Trains"
    assert "first real paragraph" in doc.text
    assert "second real paragraph" in doc.text
    assert "list item" in doc.text
    # noise should be gone
    assert "tracking pixel" not in doc.text
    assert "Site Header" not in doc.text
    assert "Copyright 2026" not in doc.text


@patch("ingest.requests.get")
def test_extract_from_url_raises_on_thin_content(mock_get):
    thin_html = "<html><head><title>Empty</title></head><body><p>Hi</p></body></html>"
    mock_get.return_value = _make_mock_response(thin_html)
    with pytest.raises(IngestError, match="enough readable text"):
        extract_from_url("https://example.com/empty")


@patch("ingest.requests.get")
def test_extract_from_url_handles_request_exception(mock_get):
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("dns failure")
    with pytest.raises(IngestError, match="Couldn't fetch"):
        extract_from_url("https://example.com/unreachable")


def _make_test_pdf_bytes(paragraphs):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for p in paragraphs:
        story.append(Paragraph(p, styles["Normal"]))
        story.append(Spacer(1, 12))
    doc.build(story)
    return buf.getvalue()


def test_extract_from_pdf_round_trip():
    paragraphs = [
        "This is a test PDF generated for unit testing the ingestion pipeline.",
        "It contains multiple paragraphs of filler text about trains and railways.",
        "The third paragraph exists purely to give the extractor enough content to pass the minimum length check.",
    ]
    pdf_bytes = _make_test_pdf_bytes(paragraphs)
    doc = extract_from_pdf(pdf_bytes, "test_document.pdf")

    assert doc.title == "test_document"
    assert doc.source_label == "test_document.pdf"
    assert "test PDF generated for unit testing" in doc.text
    assert "filler text about trains" in doc.text


def test_extract_from_pdf_raises_on_corrupted_bytes():
    with pytest.raises(IngestError, match="Couldn't read that PDF"):
        extract_from_pdf(b"this is not a real pdf file", "broken.pdf")


def test_extract_from_pdf_raises_on_thin_content():
    pdf_bytes = _make_test_pdf_bytes(["Hi"])
    with pytest.raises(IngestError, match="enough readable text"):
        extract_from_pdf(pdf_bytes, "thin.pdf")
