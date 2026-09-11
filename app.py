"""
ContextIQ — Streamlit front end.

Run with:  streamlit run app.py

Flow:
  1. Upload a PDF in the sidebar -> it's chunked, embedded, and stored in Chroma.
  2. Ask a question in the chat box -> relevant chunks are retrieved and sent
     to a local Ollama LLM, which answers grounded strictly in that context.
"""

import os
import time

import streamlit as st

import config
import rag_engine

st.set_page_config(page_title="ContextIQ", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"?}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = (
        rag_engine.load_existing_vector_store()
        if rag_engine.has_existing_index()
        else None
    )

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------------------------------------------------------------------
# Sidebar — document upload & index management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 ContextIQ")
    st.caption("Ask questions about your own PDFs — answered from their content.")

    st.divider()
    st.subheader("1. Upload a document")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    replace_index = st.checkbox(
        "Replace existing index",
        value=False,
        help="If off, this PDF is added alongside anything already indexed.",
    )

    if uploaded_file is not None:
        if st.button("Ingest PDF", type="primary", use_container_width=True):
            save_path = os.path.join(config.DATA_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Splitting, embedding, and indexing…"):
                start = time.time()
                num_chunks = rag_engine.ingest_pdf(
                    save_path, reset_first=replace_index
                )
                st.session_state.vector_store = rag_engine.load_existing_vector_store()
                elapsed = time.time() - start

            if replace_index:
                st.session_state.indexed_files = [uploaded_file.name]
            else:
                st.session_state.indexed_files.append(uploaded_file.name)

            st.success(f"Indexed {num_chunks} chunks in {elapsed:.1f}s ✅")

    st.divider()
    st.subheader("2. Indexed documents")
    if st.session_state.indexed_files:
        for fname in st.session_state.indexed_files:
            st.markdown(f"- 📄 {fname}")
    else:
        st.caption("No documents indexed yet in this session.")

    if st.button("🗑️ Clear index", use_container_width=True):
        rag_engine.reset_vector_store()
        st.session_state.vector_store = None
        st.session_state.indexed_files = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    with st.expander("⚙️ Settings"):
        st.write(f"**LLM:** {config.OLLAMA_MODEL} (via Ollama)")
        st.write(f"**Embeddings:** {config.EMBEDDING_MODEL_NAME}")
        st.write(f"**Chunk size / overlap:** {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}")
        st.write(f"**Chunks retrieved per query:** {config.TOP_K}")


# ---------------------------------------------------------------------------
# Main panel — chat interface
# ---------------------------------------------------------------------------
st.header("Chat with your documents")

if st.session_state.vector_store is None:
    st.info("👈 Upload and ingest a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View retrieved sources"):
                    for i, doc in enumerate(msg["sources"], start=1):
                        source = doc.metadata.get("source_file", "document")
                        page = doc.metadata.get("page")
                        label = f"{source}" + (
                            f", page {page + 1}" if isinstance(page, int) else ""
                        )
                        st.markdown(f"**Chunk {i} — {label}**")
                        st.text(doc.page_content[:500])

    question = st.chat_input("Ask a question about your document…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating an answer…"):
                try:
                    answer, sources = rag_engine.answer_question(
                        question, st.session_state.vector_store
                    )
                except Exception as e:
                    answer = (
                        "⚠️ Couldn't reach the local LLM. Make sure Ollama is "
                        f"running (`ollama serve`) and the model is pulled "
                        f"(`ollama pull {config.OLLAMA_MODEL}`).\n\nError: {e}"
                    )
                    sources = []
            st.markdown(answer)
            if sources:
                with st.expander("View retrieved sources"):
                    for i, doc in enumerate(sources, start=1):
                        source = doc.metadata.get("source_file", "document")
                        page = doc.metadata.get("page")
                        label = f"{source}" + (
                            f", page {page + 1}" if isinstance(page, int) else ""
                        )
                        st.markdown(f"**Chunk {i} — {label}**")
                        st.text(doc.page_content[:500])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
