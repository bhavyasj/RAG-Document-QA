"""
build_index.py
Embeds all chunks using a HuggingFace sentence-transformer model and
builds a FAISS vector index for similarity search.

Model: sentence-transformers/all-MiniLM-L6-v2
Chosen because it's small (~80MB), fast on CPU, and a standard, well-
regarded baseline for semantic search -- a sensible default for a
resource-constrained Werkstudent demo project that still needs to run
on free-tier hosting (HuggingFace Spaces CPU instances).
"""

import json
import os
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from chunking import load_chunks, build_all_chunks, CHUNKS_PATH

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(chunks, model: SentenceTransformer) -> np.ndarray:
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    # Inner product on normalized vectors == cosine similarity search.
    # Flat index is fine at this corpus scale (hundreds-to-low-thousands
    # of chunks); would swap for IVF/HNSW only at much larger scale.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks, index_dir: str = None):
    # resolve at call time (not import time) so overriding the module
    # paths -- as the tests do -- redirects every file this writes
    index_dir = index_dir or os.path.dirname(FAISS_INDEX_PATH)
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def ensure_demo_index(model=None):
    """Builds the demo FAISS index from data/raw if it isn't on disk yet.

    The index is a generated artifact (gitignored), and hosts like
    HuggingFace Spaces only install requirements and launch the app --
    they never run build_index.py. Calling this at app start makes a
    fresh clone / fresh Space work without any manual build step.
    Returns True if it had to build the index, False if it already existed.
    """
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        return False
    chunks = build_all_chunks()
    if model is None:
        model = get_embedding_model()
    index = build_faiss_index(embed_chunks(chunks, model))
    save_index(index, chunks)
    return True


def main():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run chunking.py first."
        )

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks loaded")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    print("  (downloads ~80MB on first run, requires internet access)")
    model = get_embedding_model()

    print("\nEmbedding chunks...")
    embeddings = embed_chunks(chunks, model)
    print(f"  embeddings shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"  index size: {index.ntotal} vectors, dim={index.d}")

    print(f"\nSaving index to {INDEX_DIR}")
    save_index(index, chunks)
    print("Done.")


if __name__ == "__main__":
    main()
