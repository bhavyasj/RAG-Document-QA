# RAG Document Q&A Assistant

A retrieval-augmented generation (RAG) chatbot. By default it answers
questions about German rail transport (Deutsche Bahn, ICE, S-Bahn,
Regional-Express) grounded in a small Wikipedia-based knowledge base —
but it can also ingest **any URL or uploaded PDF on the fly** and answer
questions about that content instead. A BM25 keyword-search baseline
runs alongside the semantic retriever for direct comparison on whichever
source is active.

Built as a portfolio project for Werkstudent ML/AI applications in Germany.

## Problem

Generic chatbots hallucinate facts and can't cite sources. This project
demonstrates a production-style RAG pattern: retrieve relevant passages
from a known document set, then constrain an LLM to answer only from
that retrieved context — with every answer traceable back to its source
chunks. It also demonstrates that the same pipeline generalizes beyond
one fixed corpus: point it at a news article, a blog post, or a PDF
report, and it builds a fresh index and answers questions about that
specific document instead.

## Approach

```
Source: bundled Wikipedia articles, OR a pasted URL, OR an uploaded PDF
        │
        ▼
  Text extraction (ingest.py: BeautifulSoup for URLs, pdfplumber for PDFs)
        │
        ▼
  Chunking (paragraph-aware, ~800 chars, 150-char overlap)
        │
        ▼
  Embedding (sentence-transformers/all-MiniLM-L6-v2)
        │
        ▼
  FAISS vector index (cosine similarity via normalized inner product)
        │
        ▼
  Query → top-k retrieval → LangChain prompt template → flan-t5-base
        │
        ▼
  Grounded answer + cited source chunks
```

A parallel **BM25 keyword-search retriever** runs over the same chunk
corpus, so retrieval quality can be measured and compared directly
rather than asserted — for the bundled demo corpus and for whatever
ad-hoc URL or PDF is currently loaded.

### Stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small (~80MB), fast on CPU, strong baseline for semantic search |
| Vector store | FAISS (`IndexFlatIP`) | Exact search is fine at this corpus scale; no need for IVF/HNSW |
| Generation LLM | `google/flan-t5-base` | Free, instruction-tuned, runs on CPU — fits free-tier hosting |
| Orchestration | LangChain | Standard, recruiter-recognizable RAG framework |
| Baseline | `rank-bm25` | Classic keyword search, the natural comparison point for "why RAG" |
| Ad-hoc ingestion | `requests` + `BeautifulSoup` (URLs), `pdfplumber` (PDFs) | Lightweight, dependable extraction for typical articles and text-based PDFs |
| UI | Streamlit | Fast to build, deploys free on HuggingFace Spaces |

## Results

Run `src/evaluate.py` to reproduce. It measures **accuracy@3** (does the
retriever return a chunk from the correct source document in its top 3
results?) for both retrievers across 20 hand-written test questions
spanning all 5 source articles, against the bundled demo corpus.

```
=== BM25 baseline ===
  Doc accuracy@3:       100.0%  (20/20)
  Avg keyword coverage:  90.0%

=== Semantic (FAISS + embeddings) ===
  Doc accuracy@3:        95.0%  (19/20)
  Avg keyword coverage:  97.5%

Relative result: BM25 outperformed semantic retrieval by 5% on this corpus.
```

**This is a real finding worth understanding rather than a disappointing
one to hide.** On a small corpus (5 articles, ~20 chunks) dense with
distinctive named entities — ICE, S-Bahn, Deutschlandticket, FlixTrain —
keyword search has a natural advantage: exact entity matches are easy
and there's little lexical ambiguity for embeddings to resolve. Semantic
retrieval's advantage tends to show up on larger corpora and on
paraphrased or conceptual questions where the exact keywords don't
appear in the source text. The keyword-coverage metric (97.5% vs 90.0%)
shows semantic retrieval still pulled in highly relevant content even
on the one question it missed at the document level.

**For a resume/interview, this is the honest version**: *"Built a RAG
pipeline with LangChain + FAISS and benchmarked it against a BM25
baseline across 20 test questions; found BM25 slightly outperformed
semantic retrieval on this small, entity-dense corpus — consistent with
known limitations of dense retrieval at small scale — and used that to
reason about when each approach is appropriate."* This is a stronger
interview answer than a generic "RAG is better" claim because it shows
understanding of when retrieval methods help versus when they don't.

## Project structure

```
.
├── app/
│   └── streamlit_app.py       # Chat UI with source switcher (demo / URL / PDF)
├── src/
│   ├── fetch_wikipedia.py     # Pulls live Wikipedia articles for the demo corpus
│   ├── ingest.py              # Extracts text from an arbitrary URL or uploaded PDF
│   ├── chunking.py            # Splits documents into overlapping chunks
│   ├── build_index.py         # Embeds chunks, builds FAISS index
│   ├── retrieval.py           # SemanticRetriever (FAISS) + BM25Retriever (baseline)
│   ├── rag_chain.py           # RAG chain: retrieval -> token-budgeted prompt -> LLM
│   └── evaluate.py            # Compares both retrievers on test questions (demo corpus)
├── data/
│   ├── raw/                   # Demo corpus source documents (.txt) + manifest.json
│   └── test_questions.json    # 20 Q&A pairs used for evaluation
├── tests/                     # Unit + integration tests (run offline, no model downloads)
├── .github/workflows/         # CI: runs the test suite on every push / PR
├── requirements.txt
├── requirements-dev.txt       # Adds pytest + reportlab for running the test suite
├── Dockerfile
├── DEPLOY.md                  # Step-by-step: GitHub + HuggingFace Spaces
└── LICENSE                    # MIT
```

## How to run

### 1. Setup

```bash
git clone https://github.com/bhavyasj/rag-document-qa.git
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Get documents

Sample seed articles are already included in `data/raw/` (5 short
rail-transport articles) so you can run everything immediately without
hitting the live Wikipedia API.

To fetch fresh/different articles instead:

```bash
cd src
python fetch_wikipedia.py
```

Edit the `TOPICS` list at the top of `fetch_wikipedia.py` to change the
subject domain.

### 3. Build the index

(Optional: the app builds the demo index automatically on first start if
it's missing. Run these steps manually if you want to run `evaluate.py`.)

```bash
cd src
python chunking.py      # produces data/chunks.json
python build_index.py   # downloads the embedding model, builds FAISS index
```

### 4. Evaluate retrieval quality (optional but recommended)

```bash
python evaluate.py
```

This prints accuracy@3 for both BM25 and semantic retrieval, and saves
full results to `data/evaluation_results.json`. Use the printed
relative improvement number for your resume bullet — not the
placeholder.

### 5. Run the chat app

```bash
cd ..
streamlit run app/streamlit_app.py
```

First run downloads `flan-t5-base` (~1GB) — needs internet access.

### 6. Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All 25 tests run fully offline: chunking, retrieval, ingestion (URL
fetching is mocked; PDF extraction uses a real generated test PDF), and
full pipeline wiring are verified using a deterministic fake embedder
in place of the real model. This confirms pipeline correctness, not
answer quality. Real retrieval quality is measured by `evaluate.py`
against the real model on the bundled demo corpus.

### 7. Ask questions about your own content

Once the app is running, use the sidebar to switch the knowledge
source:

- **Demo corpus** — the default, German rail-transport articles
- **A URL I paste** — paste any article/blog/news URL, click "Load this
  URL", and the app fetches the page, extracts the main text, builds a
  fresh FAISS + BM25 index for it, and clears the chat so you're
  starting fresh on the new content
- **A PDF I upload** — upload a PDF, click "Load this PDF", and the same
  ingest → chunk → index flow runs on the extracted PDF text

Switching source replaces the active knowledge base for that session
(a deliberate scope choice — see Known limitations for the
alternative "merge multiple sources" design that was considered).

## Deployment

### Docker

```bash
docker build -t rag-doc-qa .
docker run -p 8501:8501 rag-doc-qa
```

Note: `docker build` downloads both HF models during the image build
step, so it needs internet access at build time, not just at runtime.

## Known limitations

- `flan-t5-base` has a 512-token input limit, so with ~800-character chunks
  usually only the top 2 of the 3 retrieved chunks fit in the prompt. The
  chain adds chunks in rank order until the limit is reached (so the
  question is never truncated away), and the UI shows how many were used.

- `flan-t5-base` is a small model; answers are serviceable but not as
  fluent as GPT-4-class models. This was a deliberate cost/latency
  tradeoff for free CPU hosting — documented here rather than hidden,
  since being upfront about tradeoffs is itself a useful interview talking point.
- The demo knowledge base is intentionally small (5 articles, ~20 chunks)
  for a fast, inspectable demo. The same pipeline scales to thousands
  of documents by swapping `IndexFlatIP` for an approximate index
  (HNSW/IVF) at the point where exact search becomes too slow.
- Evaluation uses document-level accuracy@k and keyword coverage as
  proxies for answer quality, not a full LLM-judged answer-correctness
  score — a reasonable scope decision for a portfolio project, but a
  fair follow-up question to expect in an interview.
- **URL extraction is heuristic, not a full readability algorithm.** It
  strips known noise tags (`nav`, `header`, `footer`, scripts, etc.) and
  keeps paragraph/heading/list text in document order. This works well
  on typical articles, blogs, and Wikipedia-style pages, but can
  produce thin or noisy results on JS-heavy single-page apps, paywalled
  content, or unusually structured pages. The app surfaces a clear
  error in that case rather than silently returning garbage.
- **PDF extraction does not run OCR.** Scanned/image-only PDFs will
  fail extraction with a clear error rather than returning empty or
  garbled text. Adding OCR (e.g. via `pytesseract`) is a natural next
  step but was out of scope for this version.
- **Switching knowledge source replaces rather than merges.** Loading a
  new URL or PDF replaces the active knowledge base for the session
  instead of adding to it. This was a deliberate simplicity choice; a
  "merge sources" version would need to tag chunks by source and likely
  show per-source filtering in the UI — a reasonable extension to
  mention in an interview as a "if I had more time" answer.
- Ad-hoc ingestion is tuned for short-to-medium documents (a few pages).
  Very long PDFs (50+ pages) will still chunk and index correctly, but
  haven't been performance-tested here, and embedding a very large
  document on a free CPU instance could be slow.
