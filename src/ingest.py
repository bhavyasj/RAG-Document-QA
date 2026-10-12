"""
ingest.py
Extracts clean text from an arbitrary URL or an uploaded PDF, so the RAG
pipeline can answer questions about ad-hoc content instead of only the
bundled Wikipedia rail-transport demo corpus.

Two entry points, both returning a list of (title, source_label, text)
tuples ready to be passed into chunking.chunk_document():

    extract_from_url(url: str)
    extract_from_pdf(file_bytes: bytes, filename: str)

Design notes:
- URL extraction uses BeautifulSoup with a simple "largest text block"
  heuristic rather than a full readability algorithm. This is a
  deliberate scope tradeoff for a portfolio project: it handles typical
  article/blog/news pages well, but isn't bulletproof against every
  site layout (heavy JS-rendered pages, paywalls, and unusual markup
  can still produce thin or noisy text). Documented here and in the
  README rather than silently failing.
- PDF extraction uses pdfplumber, which preserves reading order better
  than raw PDF text streams for typical single/double-column documents.
- Both paths raise IngestError with a user-facing message on failure,
  so the Streamlit UI can show a clean error instead of a stack trace.
"""

import io
import re
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

MAX_FETCH_BYTES = 5_000_000  # 5MB cap on downloaded HTML, avoids pathological pages
REQUEST_TIMEOUT_SECONDS = 15
MIN_EXTRACTED_CHARS = 200  # below this, treat extraction as failed


class IngestError(Exception):
    """Raised when a URL or PDF can't be turned into usable text.
    Message is written to be shown directly to the end user.
    """


@dataclass
class IngestedDocument:
    title: str
    source_label: str  # what to show as the "filename"/source in citations
    text: str


def _clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Tags that are reliably non-content on most article/blog/news pages.
_NOISE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "form",
    "noscript", "iframe", "svg", "button",
]


def extract_from_url(url: str) -> IngestedDocument:
    """Fetches a URL and extracts the main readable text using a
    heuristic: strip known noise tags, then keep <p>/<h1-h3>/<li> text
    in document order. Works well on typical article/blog/Wikipedia-
    style pages; may produce thin results on JS-heavy or paywalled
    sites, which is reported back to the caller as an IngestError
    rather than returned as silently-empty text.
    """
    if not re.match(r"^https?://", url.strip(), re.IGNORECASE):
        raise IngestError(
            "That doesn't look like a valid URL. Make sure it starts with "
            "http:// or https://."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RAGDocQABot/1.0; "
            "portfolio-project; respects robots.txt)"
        )
    }

    try:
        response = requests.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, stream=True
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise IngestError(f"Timed out trying to fetch {url}. Try again or use a different link.")
    except requests.exceptions.RequestException as e:
        raise IngestError(f"Couldn't fetch that URL ({e}). Check the link and try again.")

    content = response.raw.read(MAX_FETCH_BYTES + 1, decode_content=True)
    if len(content) > MAX_FETCH_BYTES:
        raise IngestError("That page is too large to process (over 5MB of HTML).")

    soup = BeautifulSoup(content, "html.parser")

    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else url

    content_tags = soup.find_all(["p", "h1", "h2", "h3", "li"])
    paragraphs = [t.get_text(" ", strip=True) for t in content_tags]
    paragraphs = [p for p in paragraphs if len(p) > 20]  # drop short nav/menu fragments
    text = _clean_whitespace("\n\n".join(paragraphs))

    if len(text) < MIN_EXTRACTED_CHARS:
        raise IngestError(
            "Couldn't extract enough readable text from that page. It may be "
            "JS-rendered, paywalled, or not a typical article layout. Try a "
            "different URL or paste the text directly."
        )

    return IngestedDocument(title=title, source_label=url, text=text)


def extract_from_pdf(file_bytes: bytes, filename: str) -> IngestedDocument:
    """Extracts text from an uploaded PDF using pdfplumber, which
    preserves reading order reasonably well for typical single/double
    column documents. Scanned/image-only PDFs will yield little or no
    text since this does not run OCR -- that's a documented limitation,
    not a silent failure: callers get a clear IngestError instead.
    """
    import pdfplumber  # imported here to keep ingest.py importable even
                        # in contexts where pdfplumber isn't installed yet

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_texts = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_texts.append(page_text)
            text = _clean_whitespace("\n\n".join(page_texts))
    except Exception as e:
        raise IngestError(f"Couldn't read that PDF ({e}). Make sure it's not corrupted or password-protected.")

    if len(text) < MIN_EXTRACTED_CHARS:
        raise IngestError(
            "Couldn't extract enough readable text from that PDF. It may be "
            "a scanned/image-only document (this tool doesn't run OCR), or "
            "the PDF may use an unusual text encoding."
        )

    title = filename.rsplit(".", 1)[0]
    return IngestedDocument(title=title, source_label=filename, text=text)


if __name__ == "__main__":
    # Quick manual smoke test against a known-stable page.
    # Requires real internet access -- not run automatically in CI/sandbox.
    doc = extract_from_url("https://en.wikipedia.org/wiki/Deutsche_Bahn")
    print(f"Title: {doc.title}")
    print(f"Extracted {len(doc.text)} chars")
    print(doc.text[:500])
