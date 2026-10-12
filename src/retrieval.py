"""
retrieval.py
Two retrievers over the same chunk corpus:
  1. SemanticRetriever  -- FAISS + sentence-transformer embeddings (the RAG approach)
  2. BM25Retriever      -- classic keyword/term-frequency search (the baseline)

Both expose the same .retrieve(query, k) -> List[Chunk] interface so the
evaluation script can swap between them transparently.
"""

import re
from typing import List

from rank_bm25 import BM25Okapi

from build_index import load_index, get_embedding_model, EMBEDDING_MODEL_NAME
from chunking import Chunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class SemanticRetriever:
    """FAISS-based dense retrieval using sentence-transformer embeddings.

    By default loads the persistent on-disk index built by
    build_index.py (the bundled rail-transport demo corpus). Pass
    `index` and `chunks` explicitly to use an in-memory index instead --
    e.g. one built on the fly from a user-supplied URL or PDF -- without
    touching the on-disk demo index at all.
    """

    def __init__(self, model=None, index=None, chunks=None):
        if index is not None and chunks is not None:
            self.index, self.chunks = index, chunks
        else:
            self.index, self.chunks = load_index()
        self.model = model if model is not None else get_embedding_model(EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, k: int = 3) -> List[Chunk]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.chunks[idx])
        return results


class BM25Retriever:
    """Classic keyword search baseline -- no embeddings, no LLM, pure
    term-frequency ranking. This is the comparison point used to
    measure how much the RAG/semantic approach improves answer quality.
    """

    def __init__(self, chunks: List[Chunk] = None):
        if chunks is None:
            _, chunks = load_index()
        self.chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 3) -> List[Chunk]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.chunks[i] for i in top_indices]


if __name__ == "__main__":
    # Quick smoke test (requires build_index.py to have been run first,
    # and requires a real or substitute embedding model available)
    bm25 = BM25Retriever()
    results = bm25.retrieve("How fast can ICE trains go?", k=3)
    print("BM25 top results:")
    for r in results:
        print(f"  - {r.doc_title} :: {r.text[:80]}...")
