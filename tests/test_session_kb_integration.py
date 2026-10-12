"""
test_session_kb_integration.py
Simulates the core logic the Streamlit app uses to build and query a
session-scoped knowledge base from ad-hoc text (as if it came from a
URL or PDF), without actually running Streamlit. This exercises the
same chunking -> embedding -> FAISS -> retrieval -> RAG chain path that
app/streamlit_app.py's build_session_knowledge_base() uses internally,
using a fake embedder/LLM so it runs offline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_fake_embedder import FakeSentenceTransformer
from chunking import chunk_document
from build_index import embed_chunks, build_faiss_index
from retrieval import SemanticRetriever, BM25Retriever
from rag_chain import RAGChain


class FakeLLM:
    def invoke(self, prompt_text):
        return f"[fake answer, prompt had {len(prompt_text)} chars]"


SAMPLE_ARTICLE_TEXT = """A Brief History of the Eiffel Tower

The Eiffel Tower was completed in 1889 for the World's Fair held in Paris, marking the centennial of the French Revolution. It was designed by engineers working at Gustave Eiffel's company, though the original concept is often credited to Maurice Koechlin and Emile Nouguier, two senior engineers at the firm.

The tower initially faced significant criticism from a number of prominent Parisian artists and intellectuals, who considered it an eyesore that clashed with the city's classical architecture. A widely circulated open letter at the time condemned the project in strong terms, though public opinion gradually warmed to the structure over the following decades.

Standing at 330 meters including its antennas, the Eiffel Tower was the tallest man-made structure in the world for over four decades, a record it held until the completion of the Chrysler Building in New York City in 1930. It remained the tallest structure in France for many more decades after that.

The tower was originally intended to be a temporary installation, scheduled for demolition after twenty years. It was saved largely because of its usefulness for radio telegraphy, and later for scientific experiments, which gave the structure ongoing practical value beyond its role as a fairground attraction.

Today the Eiffel Tower receives several million visitors annually, making it one of the most visited paid monuments in the world. It remains one of the most recognized landmarks globally and a defining symbol of Paris and of France more broadly.
"""


def test_session_kb_can_be_built_and_queried_end_to_end():
    fake_model = FakeSentenceTransformer()

    # Mirrors build_session_knowledge_base() in streamlit_app.py
    chunks = chunk_document(
        title="A Brief History of the Eiffel Tower",
        filename="eiffel_tower_article.html",
        text=SAMPLE_ARTICLE_TEXT,
    )
    assert len(chunks) > 0

    embeddings = embed_chunks(chunks, fake_model)
    faiss_index = build_faiss_index(embeddings)

    semantic_retriever = SemanticRetriever(model=fake_model, index=faiss_index, chunks=chunks)
    bm25_retriever = BM25Retriever(chunks=chunks)

    chain = RAGChain(retriever=semantic_retriever, llm=FakeLLM(), k=2)
    result = chain.answer("When was the Eiffel Tower completed?")

    assert "answer" in result
    # this short sample article only produces 1 chunk, so even with k=2
    # requested, retrieval correctly returns just the 1 available chunk
    # rather than padding with anything else -- that's correct behavior,
    # not a bug, and worth asserting explicitly here.
    assert len(result["sources"]) == min(2, len(chunks))
    assert len(result["sources"]) >= 1
    # all sources should come from the session doc, not any other corpus
    assert all(s["title"] == "A Brief History of the Eiffel Tower" for s in result["sources"])

    # BM25 baseline should also work standalone against the same chunks
    bm25_results = bm25_retriever.retrieve("Eiffel Tower completed year", k=2)
    assert len(bm25_results) == min(2, len(chunks))


def test_session_kb_does_not_leak_into_demo_corpus_chunks():
    """Guards against accidentally mixing session-scoped chunks with
    the bundled demo corpus -- the chunks used to build the session
    index should only ever be the ones derived from the ad-hoc text.
    """
    fake_model = FakeSentenceTransformer()
    chunks = chunk_document(
        title="A Brief History of the Eiffel Tower",
        filename="eiffel_tower_article.html",
        text=SAMPLE_ARTICLE_TEXT,
    )
    embeddings = embed_chunks(chunks, fake_model)
    faiss_index = build_faiss_index(embeddings)
    retriever = SemanticRetriever(model=fake_model, index=faiss_index, chunks=chunks)

    results = retriever.retrieve("Eiffel Tower height", k=5)
    assert all(r.source_file == "eiffel_tower_article.html" for r in results)
    assert all("Deutsche" not in r.text and "ICE" not in r.text for r in results)
