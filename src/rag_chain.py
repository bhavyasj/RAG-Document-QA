"""
rag_chain.py
Wires the SemanticRetriever into a LangChain pipeline that retrieves
relevant chunks and feeds them to a local, free HuggingFace text-
generation model to produce a grounded answer.

Model choice: google/flan-t5-base
Chosen because it's a free, CPU-friendly instruction-tuned model (no API
key, no GPU required) that's reliable at the kind of short extractive/
abstractive QA this project needs, unlike base GPT-2-style models which
aren't instruction-tuned and tend to ramble or ignore the prompt. Larger
HF models (Mistral-7B, Llama-3-8B, etc.) would give noticeably better
answers but need a GPU to run at usable speed -- flan-t5-base is the
right tradeoff for free-tier HuggingFace Spaces CPU hosting.
"""

import os
from dataclasses import replace

from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retrieval import SemanticRetriever

# Override without editing code, e.g.  set RAG_LLM_MODEL=google/flan-t5-large
LLM_MODEL_NAME = os.environ.get("RAG_LLM_MODEL", "google/flan-t5-base")

PROMPT_TEMPLATE = """Answer the question using only the context below. \
Reply in one or two complete sentences. \
If the context does not contain the answer, say "I don't have enough \
information to answer that."

Context:
{context}

Question: {question}

Answer:"""


class Seq2SeqLLM:
    """Thin wrapper around a local HuggingFace seq2seq model (e.g. T5/
    flan-t5) exposing the same .invoke(text) -> str interface our
    RAGChain expects from an LLM.

    Built with AutoModelForSeq2SeqLM + AutoTokenizer directly rather
    than transformers.pipeline("text2text-generation", ...) because
    that pipeline task name was removed in transformers v5 (it's still
    referenced in some docs, but raises
    `KeyError: Unknown task text2text-generation` at runtime on v5).
    Loading the model class directly is also the officially recommended
    approach for T5-family models regardless of pipeline availability.
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME, max_new_tokens: int = 200,
                 max_input_tokens: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        # flan-t5 inputs are truncated at this many tokens. RAGChain reads
        # it so it can drop retrieved chunks that don't fit, instead of
        # letting the tokenizer silently cut off the END of the prompt
        # (which is where the question and "Answer:" live).
        self.max_input_tokens = max_input_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text).input_ids)

    def invoke(self, prompt_text: str) -> str:
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=self.max_input_tokens
        )
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            # beam search gives more complete, less clipped answers than
            # greedy decoding on small flan-t5 models
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def build_llm(model_name: str = LLM_MODEL_NAME, max_new_tokens: int = 200) -> Seq2SeqLLM:
    """Loads a local HuggingFace seq2seq model for use inside RAGChain.
    Downloads ~1GB on first run; requires internet access to
    huggingface.co.
    """
    return Seq2SeqLLM(model_name=model_name, max_new_tokens=max_new_tokens)


class RAGChain:
    def __init__(self, retriever: SemanticRetriever = None, llm=None, k: int = 3):
        self.retriever = retriever if retriever is not None else SemanticRetriever()
        self.llm = llm if llm is not None else build_llm()
        self.k = k
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE,
        )

    def _render(self, question: str, chunks) -> str:
        context = "\n\n".join(f"[{c.doc_title}] {c.text}" for c in chunks)
        return self.prompt.format(context=context, question=question)

    def _build_prompt(self, question: str, chunks):
        """Returns (prompt_text, chunks_actually_used).

        If the LLM exposes `max_input_tokens` and `count_tokens`, add
        retrieved chunks in rank order, skipping any that would push the
        prompt over the limit, so the question is never truncated away.
        If even the top chunk alone is too long, it is trimmed. LLMs
        without those attributes (e.g. test fakes) get every chunk.
        """
        limit = getattr(self.llm, "max_input_tokens", None)
        count = getattr(self.llm, "count_tokens", None)
        if not limit or count is None:
            return self._render(question, chunks), list(chunks)

        used = []
        for c in chunks:
            if count(self._render(question, used + [c])) <= limit:
                used.append(c)

        if not used and chunks:
            top = chunks[0]
            text = top.text
            while text and count(self._render(question, [replace(top, text=text)])) > limit:
                text = text[: int(len(text) * 0.9)]
            used = [replace(top, text=text)]

        return self._render(question, used), used

    def answer(self, question: str):
        retrieved_chunks = self.retriever.retrieve(question, k=self.k)
        prompt_text, used_chunks = self._build_prompt(question, retrieved_chunks)
        raw_answer = self.llm.invoke(prompt_text)
        return {
            "question": question,
            "answer": raw_answer.strip(),
            "num_retrieved": len(retrieved_chunks),
            # only the chunks the model actually saw, so citations are honest
            "sources": [
                {"title": c.doc_title, "chunk_id": c.chunk_id, "text": c.text}
                for c in used_chunks
            ],
        }


if __name__ == "__main__":
    chain = RAGChain()
    result = chain.answer("What is the Deutschlandticket and what does it cover?")
    print("Q:", result["question"])
    print("A:", result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['title']} ({s['chunk_id']})")
