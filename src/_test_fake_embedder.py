"""
_test_fake_embedder.py
TEST-ONLY UTILITY -- not used by the app.

A deterministic, hash-based bag-of-words stand-in for the real
sentence-transformer model. It lets the test suite exercise the whole
pipeline (chunking -> indexing -> retrieval -> RAG chain -> UI wiring)
offline, with no model download, e.g. in CI. Its vectors are NOT
semantically meaningful, so it verifies plumbing only; real retrieval
quality is measured by evaluate.py with the real embedding model.
"""

import hashlib
import re

import numpy as np

VOCAB_DIM = 384  # match all-MiniLM-L6-v2 output dim for drop-in compatibility


def _tokenize(text: str):
    return re.findall(r"[a-zA-Z]+", text.lower())


class FakeSentenceTransformer:
    """Drop-in stand-in for sentence_transformers.SentenceTransformer
    for local plumbing tests only. Embeds via hashed token bag-of-words
    so that lexically similar texts get cosine-similar vectors -- enough
    to sanity check retrieval ranking logic without a real model.
    """

    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True,
               normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.zeros((len(texts), VOCAB_DIM), dtype="float32")
        for i, text in enumerate(texts):
            for tok in _tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % VOCAB_DIM
                sign = 1.0 if (h // VOCAB_DIM) % 2 == 0 else -1.0
                vecs[i, idx] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs
