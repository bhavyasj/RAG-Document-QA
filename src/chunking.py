"""
chunking.py
Splits raw documents into overlapping chunks suitable for embedding and
retrieval. Chunking strategy: split on paragraph boundaries first, then
greedily pack paragraphs into chunks up to a target size, with a sliding
overlap so context isn't lost at chunk boundaries.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.json")

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    source_file: str
    text: str


def load_raw_documents(raw_dir: str = RAW_DIR):
    docs = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(raw_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        title = text.split("\n", 1)[0].strip()
        docs.append({"title": title, "filename": fname, "text": text})
    return docs


def split_into_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def chunk_document(title: str, filename: str, text: str,
                    chunk_size: int = CHUNK_SIZE_CHARS,
                    overlap: int = CHUNK_OVERLAP_CHARS) -> List[Chunk]:
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk; carry overlap from the tail of the previous chunk
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n{para}".strip()
            # if a single paragraph is itself longer than chunk_size, hard-split it
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]

    if current:
        chunks.append(current)

    result = []
    for i, chunk_text in enumerate(chunks):
        result.append(
            Chunk(
                chunk_id=f"{filename}::chunk{i}",
                doc_title=title,
                source_file=filename,
                text=chunk_text,
            )
        )
    return result


def build_all_chunks(raw_dir: str = RAW_DIR) -> List[Chunk]:
    docs = load_raw_documents(raw_dir)
    all_chunks = []
    for doc in docs:
        doc_chunks = chunk_document(doc["title"], doc["filename"], doc["text"])
        all_chunks.extend(doc_chunks)
        print(f"  {doc['filename']}: {len(doc_chunks)} chunks")
    return all_chunks


def save_chunks(chunks: List[Chunk], path: str = CHUNKS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)


def load_chunks(path: str = CHUNKS_PATH) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Chunk(**r) for r in raw]


if __name__ == "__main__":
    print("Chunking documents...")
    chunks = build_all_chunks()
    save_chunks(chunks)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved to {CHUNKS_PATH}")
