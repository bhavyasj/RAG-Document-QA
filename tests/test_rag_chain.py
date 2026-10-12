"""
test_rag_chain.py
Verifies RAGChain never lets the prompt exceed the LLM's input limit and
never loses the question -- the failure mode where the tokenizer
truncates the END of an over-long prompt.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import Chunk
from rag_chain import RAGChain


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, k=3):
        return self.chunks[:k]


class LimitedFakeLLM:
    """Counts words as tokens and records the prompt it was given."""

    def __init__(self, max_input_tokens):
        self.max_input_tokens = max_input_tokens
        self.last_prompt = None

    def count_tokens(self, text):
        return len(text.split())

    def invoke(self, prompt_text):
        self.last_prompt = prompt_text
        return "ok"


def _chunk(i, n_words):
    return Chunk(chunk_id=f"f::{i}", doc_title=f"Doc{i}", source_file="f",
                 text=" ".join(f"w{i}x{j}" for j in range(n_words)))


def test_prompt_stays_within_limit_and_keeps_question():
    chunks = [_chunk(i, 100) for i in range(3)]
    llm = LimitedFakeLLM(max_input_tokens=250)
    chain = RAGChain(retriever=FakeRetriever(chunks), llm=llm, k=3)
    result = chain.answer("What is the unique question marker?")

    assert llm.count_tokens(llm.last_prompt) <= 250
    assert llm.last_prompt.rstrip().endswith("Answer:")
    assert "What is the unique question marker?" in llm.last_prompt
    assert result["num_retrieved"] == 3
    assert 0 < len(result["sources"]) < 3  # some chunks dropped, none truncated mid-way
    for s in result["sources"]:
        assert s["text"] in llm.last_prompt  # sources == what the model saw


def test_oversized_top_chunk_is_trimmed_not_dropped():
    chunks = [_chunk(0, 500)]
    llm = LimitedFakeLLM(max_input_tokens=200)
    chain = RAGChain(retriever=FakeRetriever(chunks), llm=llm, k=1)
    result = chain.answer("Question here?")

    assert llm.count_tokens(llm.last_prompt) <= 200
    assert "Question here?" in llm.last_prompt
    assert len(result["sources"]) == 1


def test_llm_without_limit_gets_all_chunks():
    class Plain:
        def invoke(self, p):
            return "x"
    chunks = [_chunk(i, 100) for i in range(3)]
    result = RAGChain(retriever=FakeRetriever(chunks), llm=Plain(), k=3).answer("q?")
    assert len(result["sources"]) == 3
