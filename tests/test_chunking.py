"""
test_chunking.py
Unit tests for the chunking logic. Run with: pytest tests/test_chunking.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_document, split_into_paragraphs


def test_split_into_paragraphs_strips_blank_lines():
    text = "Title\n\nPara one.\n\nPara two.\n\n\nPara three."
    paras = split_into_paragraphs(text)
    assert paras == ["Title", "Para one.", "Para two.", "Para three."]


def test_chunk_document_respects_size_limit():
    text = "\n".join([f"This is paragraph number {i} with some filler text." for i in range(50)])
    chunks = chunk_document("Test Doc", "test.txt", text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack since paragraphs are atomic units when small enough
        assert len(c.text) <= 260


def test_chunk_document_preserves_all_content_keywords():
    text = "Alpha keyword.\nBeta keyword.\nGamma keyword.\nDelta keyword."
    chunks = chunk_document("Test Doc", "test.txt", text, chunk_size=1000, overlap=10)
    combined = " ".join(c.text for c in chunks)
    for kw in ["Alpha", "Beta", "Gamma", "Delta"]:
        assert kw in combined


def test_chunk_document_assigns_unique_ids():
    text = "\n".join([f"Paragraph {i}." for i in range(20)])
    chunks = chunk_document("Test Doc", "test.txt", text, chunk_size=50, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_chunk_document_handles_empty_text():
    chunks = chunk_document("Empty Doc", "empty.txt", "")
    assert chunks == []


def test_chunk_document_handles_single_oversized_paragraph():
    # A single paragraph longer than chunk_size must still be hard-split
    long_para = "word " * 500  # ~2500 chars
    chunks = chunk_document("Big Doc", "big.txt", long_para, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 350  # small slack for overlap stitching
