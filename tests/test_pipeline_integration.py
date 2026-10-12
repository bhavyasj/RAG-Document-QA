"""
test_pipeline_integration.py
End-to-end pipeline test (chunking -> embedding -> FAISS index ->
retrieval -> RAG chain wiring) using a deterministic fake embedder and
fake LLM, so this can run anywhere with no internet access and no
downloaded model weights -- useful for CI and for quickly verifying
nothing is broken after a code change.

This test intentionally does NOT assert anything about answer quality
or real semantic relevance (the fake embedder isn't semantically
meaningful) -- it only verifies that data flows correctly through every
stage without errors and that the output shape is correct. Real quality
evaluation happens in evaluate.py against the real embedding model.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_fake_embedder import FakeSentenceTransformer
from chunking import build_all_chunks
from build_index import embed_chunks, build_faiss_index, save_index, load_index
from retrieval import SemanticRetriever
from rag_chain import RAGChain


class FakeLLM:
    def invoke(self, prompt_text):
        return f"[fake answer based on {len(prompt_text)} chars of prompt]"


def test_full_pipeline_runs_without_error(tmp_path, monkeypatch):
    import build_index as bi

    # redirect index output to a temp dir so this test doesn't touch
    # the real data/index used by the shipped app
    monkeypatch.setattr(bi, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(bi, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(bi, "METADATA_PATH", str(tmp_path / "metadata.pkl"))

    chunks = build_all_chunks()
    assert len(chunks) > 0

    fake_model = FakeSentenceTransformer()
    embeddings = embed_chunks(chunks, fake_model)
    assert embeddings.shape[0] == len(chunks)

    index = build_faiss_index(embeddings)
    assert index.ntotal == len(chunks)

    bi.save_index(index, chunks)
    loaded_index, loaded_chunks = bi.load_index()
    assert loaded_index.ntotal == len(chunks)
    assert len(loaded_chunks) == len(chunks)

    import retrieval as rt
    monkeypatch.setattr(rt, "load_index", bi.load_index)

    retriever = SemanticRetriever(model=fake_model)
    results = retriever.retrieve("test query about trains", k=3)
    assert len(results) == 3

    chain = RAGChain(retriever=retriever, llm=FakeLLM(), k=2)
    result = chain.answer("What is the Deutschlandticket?")
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 2
    assert all("title" in s and "text" in s for s in result["sources"])


def test_ensure_demo_index_builds_when_missing_and_is_idempotent(tmp_path, monkeypatch):
    import build_index as bi

    monkeypatch.setattr(bi, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(bi, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(bi, "METADATA_PATH", str(tmp_path / "metadata.pkl"))

    model = FakeSentenceTransformer()
    assert bi.ensure_demo_index(model) is True    # built from data/raw
    assert (tmp_path / "faiss.index").exists()
    assert bi.ensure_demo_index(model) is False   # already there -> no rebuild
    index, chunks = bi.load_index()
    assert index.ntotal == len(chunks) > 0
