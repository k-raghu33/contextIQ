"""
ContextIQ — central configuration.

Keeping every tunable knob in one place makes it trivial to swap models,
chunk sizes, or storage locations without hunting through the codebase.
"""

import os

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")            # uploaded PDFs land here
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "chroma_db")  # persisted Chroma DB

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --- Text splitting ------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Embedding model -------------------------------------------------------
# Runs locally via sentence-transformers, no API key needed.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Local LLM (via Ollama) ------------------------------------------------
# Make sure you've pulled this model: `ollama pull llama3.1`
OLLAMA_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.2

# --- Retrieval ---------------------------------------------------------
TOP_K = 4  # number of chunks pulled from the vector store per query

# --- Chroma collection ---------------------------------------------------
COLLECTION_NAME = "contextiq_docs"
