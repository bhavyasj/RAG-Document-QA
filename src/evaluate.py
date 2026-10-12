"""
evaluate.py
Compares the semantic (FAISS + embeddings) retriever against the BM25
keyword-search baseline on a fixed set of test questions.

Metric: retrieval accuracy@k -- for each question, did the retriever
return at least one chunk from the expected source document within its
top-k results? This is a standard, easy-to-defend retrieval metric for
a portfolio project (cleaner to reason about than end-to-end generation
quality, which is harder to score automatically without a judge model).

A secondary keyword-coverage metric is also reported: of the
expected_keywords listed for each question, what fraction appear
(case-insensitive) somewhere in the retrieved chunks' text. This gives
a rougher proxy for "would the LLM have had the right facts available
to answer correctly."

Run after build_index.py has produced data/index/.
"""

import json
import os

from retrieval import SemanticRetriever, BM25Retriever

TEST_QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "test_questions.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "evaluation_results.json"
)

K = 3


def load_test_questions():
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retriever(retriever, test_questions, k: int = K, name: str = ""):
    hits = 0
    keyword_scores = []
    per_question_results = []

    for item in test_questions:
        retrieved = retriever.retrieve(item["question"], k=k)
        retrieved_files = {c.source_file for c in retrieved}
        is_hit = item["expected_doc"] in retrieved_files
        hits += int(is_hit)

        combined_text = " ".join(c.text for c in retrieved).lower()
        kw_hits = sum(
            1 for kw in item["expected_keywords"] if kw.lower() in combined_text
        )
        kw_score = kw_hits / len(item["expected_keywords"]) if item["expected_keywords"] else 0
        keyword_scores.append(kw_score)

        per_question_results.append(
            {
                "question": item["question"],
                "expected_doc": item["expected_doc"],
                "retrieved_docs": list(retrieved_files),
                "doc_hit": is_hit,
                "keyword_coverage": round(kw_score, 2),
            }
        )

    accuracy_at_k = hits / len(test_questions)
    avg_keyword_coverage = sum(keyword_scores) / len(keyword_scores)

    print(f"\n=== {name} ===")
    print(f"  Doc accuracy@{k}:      {accuracy_at_k:.1%}  ({hits}/{len(test_questions)})")
    print(f"  Avg keyword coverage: {avg_keyword_coverage:.1%}")

    return {
        "name": name,
        "k": k,
        "accuracy_at_k": accuracy_at_k,
        "avg_keyword_coverage": avg_keyword_coverage,
        "hits": hits,
        "total": len(test_questions),
        "per_question": per_question_results,
    }


def main():
    test_questions = load_test_questions()
    print(f"Loaded {len(test_questions)} test questions")

    bm25 = BM25Retriever()
    bm25_results = evaluate_retriever(bm25, test_questions, k=K, name="BM25 baseline")

    semantic = SemanticRetriever()
    semantic_results = evaluate_retriever(semantic, test_questions, k=K, name="Semantic (FAISS + embeddings)")

    improvement = (
        (semantic_results["accuracy_at_k"] - bm25_results["accuracy_at_k"])
        / bm25_results["accuracy_at_k"] * 100
        if bm25_results["accuracy_at_k"] > 0 else float("inf")
    )

    print("\n=== Summary ===")
    print(f"  BM25 accuracy@{K}:     {bm25_results['accuracy_at_k']:.1%}")
    print(f"  Semantic accuracy@{K}: {semantic_results['accuracy_at_k']:.1%}")
    print(f"  Relative improvement: {improvement:+.1f}%")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bm25": bm25_results,
                "semantic": semantic_results,
                "relative_improvement_pct": improvement,
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
