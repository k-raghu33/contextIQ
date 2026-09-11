"""
ContextIQ — core RAG engine.

Pipeline:
    PDF  ->  PyPDFLoader  ->  RecursiveCharacterTextSplitter  ->
    HuggingFace embeddings  ->  Chroma vector store  ->
    similarity-search retriever  ->  Ollama LLM  ->  grounded answer

Every function here is intentionally small and self-contained so the
Streamlit UI (app.py) stays thin — it just calls into this module.
"""

import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config


# ---------------------------------------------------------------------------
# Embeddings (loaded once, reused everywhere)
# ---------------------------------------------------------------------------
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily instantiate and cache the embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Ingestion: PDF -> chunks -> vector store
# ---------------------------------------------------------------------------
def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF into LangChain Document objects (one per page)."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Break pages into overlapping chunks sized for retrieval + context window."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], persist: bool = True) -> Chroma:
    """Embed chunks and store them in a (persisted) Chroma collection."""
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTOR_STORE_DIR if persist else None,
    )
    return vector_store


def load_existing_vector_store() -> Chroma:
    """Reopen a previously persisted Chroma collection without re-embedding."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.VECTOR_STORE_DIR,
    )


def has_existing_index() -> bool:
    """Check whether a persisted vector store already has content."""
    if not os.path.isdir(config.VECTOR_STORE_DIR):
        return False
    try:
        store = load_existing_vector_store()
        return store._collection.count() > 0
    except Exception:
        return False


def reset_vector_store() -> None:
    """Wipe the persisted index — used when the user wants to start fresh."""
    if os.path.isdir(config.VECTOR_STORE_DIR):
        shutil.rmtree(config.VECTOR_STORE_DIR)
    os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)


def ingest_pdf(file_path: str, reset_first: bool = False) -> int:
    """
    Full ingestion pipeline for one PDF.

    Returns the number of chunks that were embedded and stored.
    """
    if reset_first:
        reset_vector_store()

    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    # Tag each chunk with the source filename so answers can cite it.
    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    build_vector_store(chunks, persist=True)
    return len(chunks)


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are ContextIQ, a careful assistant that answers questions using
ONLY the context provided below, which was retrieved from documents the
user uploaded.

Rules:
- Base your answer strictly on the context. Do not use outside knowledge.
- If the context does not contain the answer, say so clearly instead of guessing.
- Be concise and direct. Use bullet points if it improves clarity.
- When helpful, mention which part of the document the info came from.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context block, labeled by source/page."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "document")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        blocks.append(f"[Chunk {i} — {source}{page_label}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def get_llm() -> ChatOllama:
    """Instantiate the local Ollama chat model."""
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
    )


def answer_question(question: str, vector_store: Chroma, k: int = None):
    """
    Retrieve relevant chunks for `question` and generate a grounded answer.

    Returns (answer_text, list_of_source_documents) so the UI can show
    both the answer and the exact chunks it was built from.
    """
    k = k or config.TOP_K
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    source_docs = retriever.invoke(question)

    llm = get_llm()

    chain = (
        {
            "context": lambda x: _format_docs(source_docs),
            "question": RunnablePassthrough(),
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer, source_docs
