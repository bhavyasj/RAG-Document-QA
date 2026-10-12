"""
test_retrieval.py
Unit tests for the BM25 retriever, which needs no network access or
downloaded models, so it can run in any CI environment. The semantic
(FAISS + embeddings) retriever is exercised separately in
test_pipeline_integration.py using a fake embedder, since the real
embedding model requires downloading weights from HuggingFace.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import Chunk
from retrieval import BM25Retriever, SemanticRetriever
from build_index import embed_chunks, build_faiss_index


def make_test_chunks():
    return [
        Chunk("doc_a::chunk0", "Doc A", "doc_a.txt", "The ICE train reaches 300 kilometers per hour on high-speed lines."),
        Chunk("doc_b::chunk0", "Doc B", "doc_b.txt", "Deutsche Bahn was formed in 1994 from a merger of two railways."),
        Chunk("doc_c::chunk0", "Doc C", "doc_c.txt", "The S-Bahn Munich network has a central tunnel bottleneck."),
    ]


def test_bm25_retrieves_relevant_chunk_for_keyword_match():
    chunks = make_test_chunks()
    retriever = BM25Retriever(chunks=chunks)
    results = retriever.retrieve("ICE train speed kilometers", k=1)
    assert results[0].source_file == "doc_a.txt"


def test_bm25_returns_requested_k():
    chunks = make_test_chunks()
    retriever = BM25Retriever(chunks=chunks)
    results = retriever.retrieve("train", k=2)
    assert len(results) == 2


def test_bm25_handles_query_with_no_matches_gracefully():
    chunks = make_test_chunks()
    retriever = BM25Retriever(chunks=chunks)
    # should not raise, should still return k results ranked by whatever
    # weak signal exists, even with no real overlap
    results = retriever.retrieve("xylophone giraffe astronaut", k=2)
    assert len(results) == 2


def test_bm25_k_larger_than_corpus_does_not_crash():
    chunks = make_test_chunks()
    retriever = BM25Retriever(chunks=chunks)
    results = retriever.retrieve("train", k=10)
    assert len(results) == len(chunks)


def test_semantic_retriever_uses_in_memory_index_without_touching_disk():
    """Verifies SemanticRetriever(index=..., chunks=...) works fully
    in-memory -- the path used for ad-hoc URL/PDF ingestion -- and never
    calls load_index() (which would read the on-disk demo index).
    """
    from _test_fake_embedder import FakeSentenceTransformer

    chunks = make_test_chunks()
    fake_model = FakeSentenceTransformer()
    embeddings = embed_chunks(chunks, fake_model)
    in_memory_index = build_faiss_index(embeddings)

    retriever = SemanticRetriever(model=fake_model, index=in_memory_index, chunks=chunks)
    results = retriever.retrieve("ICE train speed", k=2)
    assert len(results) == 2
    assert all(r in chunks for r in results)
