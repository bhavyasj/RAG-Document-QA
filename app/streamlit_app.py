"""
streamlit_app.py
Chat-style front-end for the RAG Document Q&A Assistant.

Supports three knowledge sources, selectable in the sidebar:
  1. Demo corpus    -- the bundled German rail-transport Wikipedia articles
  2. Paste a URL     -- fetches and indexes any article/blog/news page on the fly
  3. Upload a PDF     -- extracts and indexes an uploaded PDF on the fly

Switching source or ingesting new content replaces the active knowledge
base for the session (kept simple on purpose -- see project README for
the alternative "merge sources" design that was considered and skipped).

Run with: streamlit run app/streamlit_app.py
(Run from the project root so the src/ imports resolve correctly, or see
the path-fix line below.)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st

from retrieval import SemanticRetriever, BM25Retriever
from rag_chain import RAGChain, build_llm
from build_index import (
    get_embedding_model, EMBEDDING_MODEL_NAME, embed_chunks, build_faiss_index,
    ensure_demo_index,
)
from chunking import chunk_document, build_all_chunks
from ingest import extract_from_url, extract_from_pdf, IngestError

st.set_page_config(page_title="RAG Document Q&A Assistant", page_icon="🚆", layout="centered")


# ---------------------------------------------------------------------
# Cached, expensive-to-load resources (models). These do NOT depend on
# which knowledge source is active, so they're cached once per process
# regardless of source switches.
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return get_embedding_model(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading local language model (first run downloads ~1GB)...")
def load_llm():
    return build_llm()


@st.cache_resource(show_spinner="Loading demo knowledge base (FAISS index)...")
def load_demo_retriever():
    """The original bundled rail-transport demo, loaded from the
    persistent on-disk index built by build_index.py.
    """
    model = load_embedding_model()
    ensure_demo_index(model)  # first start on a fresh clone / Space builds it
    return SemanticRetriever(model=model)


@st.cache_resource(show_spinner="Loading demo BM25 baseline...")
def load_demo_bm25_retriever():
    ensure_demo_index(load_embedding_model())
    return BM25Retriever()


# ---------------------------------------------------------------------
# Session-scoped knowledge base: built on the fly from a URL or PDF,
# kept in st.session_state rather than st.cache_resource since it's
# specific to one user's session, not shared/global like the demo corpus.
# ---------------------------------------------------------------------

def build_session_knowledge_base(title: str, source_label: str, text: str):
    """Chunks + embeds the given text and builds an in-memory FAISS
    index plus a matching BM25 index, stored in session state. Does NOT
    touch the on-disk demo index.
    """
    model = load_embedding_model()
    chunks = chunk_document(title=title, filename=source_label, text=text)
    if not chunks:
        raise IngestError("That content produced no usable chunks after processing.")

    embeddings = embed_chunks(chunks, model)
    faiss_index = build_faiss_index(embeddings)

    semantic_retriever = SemanticRetriever(model=model, index=faiss_index, chunks=chunks)
    bm25_retriever = BM25Retriever(chunks=chunks)

    st.session_state.active_source = {
        "kind": "session",
        "title": title,
        "source_label": source_label,
        "num_chunks": len(chunks),
    }
    st.session_state.session_semantic_retriever = semantic_retriever
    st.session_state.session_bm25_retriever = bm25_retriever
    # any chat history belongs to the previous knowledge base
    st.session_state.messages = []


def get_active_retrievers():
    """Returns (semantic_retriever, bm25_retriever, source_info) for
    whichever knowledge base is currently active: the bundled demo
    corpus by default, or a session-scoped URL/PDF index if one has
    been built.
    """
    active = st.session_state.get("active_source", {"kind": "demo"})
    if active["kind"] == "session":
        return (
            st.session_state.session_semantic_retriever,
            st.session_state.session_bm25_retriever,
            active,
        )
    return (
        load_demo_retriever(),
        load_demo_bm25_retriever(),
        {"kind": "demo", "title": "German Rail Transport (demo corpus)", "source_label": "bundled Wikipedia articles"},
    )


def switch_to_demo():
    st.session_state.active_source = {"kind": "demo"}
    st.session_state.pop("session_semantic_retriever", None)
    st.session_state.pop("session_bm25_retriever", None)
    st.session_state.messages = []


# ---------------------------------------------------------------------
# Sidebar: source selection
# ---------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("Knowledge source")

        source_mode = st.radio(
            "Ask questions about:",
            ["Demo corpus (German rail)", "A URL I paste", "A PDF I upload"],
            index=0,
            key="source_mode_radio",
        )

        if source_mode == "Demo corpus (German rail)":
            if st.session_state.get("active_source", {}).get("kind") != "demo":
                switch_to_demo()
            st.caption("Bundled Wikipedia articles on Deutsche Bahn, ICE, S-Bahn, and Regional-Express.")

        elif source_mode == "A URL I paste":
            url = st.text_input("Article URL", placeholder="https://example.com/article")
            if st.button("Load this URL", disabled=not url):
                with st.spinner(f"Fetching and indexing {url}..."):
                    try:
                        doc = extract_from_url(url)
                        build_session_knowledge_base(doc.title, doc.source_label, doc.text)
                        st.success(f"Loaded \"{doc.title}\" ({st.session_state.active_source['num_chunks']} chunks)")
                    except IngestError as e:
                        st.error(str(e))

        elif source_mode == "A PDF I upload":
            uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
            if uploaded_file is not None and st.button("Load this PDF"):
                with st.spinner(f"Extracting and indexing {uploaded_file.name}..."):
                    try:
                        doc = extract_from_pdf(uploaded_file.read(), uploaded_file.name)
                        build_session_knowledge_base(doc.title, doc.source_label, doc.text)
                        st.success(f"Loaded \"{doc.title}\" ({st.session_state.active_source['num_chunks']} chunks)")
                    except IngestError as e:
                        st.error(str(e))

        active = st.session_state.get("active_source", {"kind": "demo"})
        if active["kind"] == "session":
            st.markdown("---")
            st.caption(f"📄 Currently active: **{active['title']}**")
            if st.button("⬅ Back to demo corpus"):
                switch_to_demo()
                st.rerun()

        st.markdown("---")
        st.header("About this project")
        st.markdown(
            "- **Retrieval:** FAISS + `all-MiniLM-L6-v2` sentence embeddings\n"
            "- **Generation:** `google/flan-t5-base` (local, free, no API key)\n"
            "- **Framework:** LangChain\n"
            "- **Baseline:** BM25 keyword search (shown alongside for comparison)\n"
        )
        st.markdown("---")
        show_baseline = st.checkbox("Show BM25 baseline comparison", value=True)
        show_sources = st.checkbox("Show retrieved source chunks", value=True)
        st.markdown("---")
        st.markdown(
            "[GitHub repo](#) · [Live demo](#)"
        )
        return show_baseline, show_sources


def main():
    st.title("🚆 RAG Document Q&A Assistant")

    active = st.session_state.get("active_source", {"kind": "demo"})
    if active["kind"] == "session":
        st.caption(
            f"Ask questions about **{active['title']}**. Answers are generated "
            "using retrieval-augmented generation over the content you provided."
        )
    else:
        st.caption(
            "Ask questions about German rail transport (Deutsche Bahn, ICE, "
            "S-Bahn, Regional-Express) — or switch to your own URL or PDF in "
            "the sidebar. Answers are generated using retrieval-augmented "
            "generation, not a general chatbot."
        )

    show_baseline, show_sources = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                semantic_retriever, bm25_retriever, source_info = get_active_retrievers()
                llm = load_llm()
                chain = RAGChain(retriever=semantic_retriever, llm=llm, k=3)
                result = chain.answer(question)

            st.markdown(result["answer"])

            if show_sources:
                with st.expander(
                    f"📄 Sources used ({len(result['sources'])} of "
                    f"{result['num_retrieved']} retrieved chunks fit the model's input limit)"
                    if len(result['sources']) < result['num_retrieved']
                    else f"📄 Sources used ({len(result['sources'])} chunks)"
                ):
                    for s in result["sources"]:
                        st.markdown(f"**{s['title']}**")
                        st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

            if show_baseline:
                with st.expander("🔍 BM25 keyword-search baseline (for comparison)"):
                    bm25_results = bm25_retriever.retrieve(question, k=3)
                    st.caption(
                        "What a plain keyword search would have retrieved for "
                        "the same question — no embeddings, no LLM:"
                    )
                    for r in bm25_results:
                        st.markdown(f"**{r.doc_title}**")
                        st.caption(r.text[:300] + ("..." if len(r.text) > 300 else ""))

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    if not st.session_state.messages:
        if active["kind"] == "session":
            st.info(f"Ask anything about \"{active['title']}\" — try a summary question first, e.g. *\"What is this document about?\"*")
        else:
            st.info(
                "Try asking: *\"What is the Deutschlandticket and what does it cover?\"* or "
                "*\"What is the top speed of ICE trains?\"*"
            )


if __name__ == "__main__":
    main()
