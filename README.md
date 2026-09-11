# 📚 ContextIQ

A local, private PDF question-answering app built with **Retrieval-Augmented
Generation (RAG)**. Upload a PDF, ask questions in plain English, and get
answers grounded strictly in the document's content — with the exact source
chunks shown alongside every answer.

Everything runs locally: embeddings via Hugging Face, vector storage via
ChromaDB, and the LLM via Ollama. No API keys, no data leaving your machine.

## Architecture

```
   PDF file
      │
      ▼
 PDF Loader (PyPDFLoader)
      │
      ▼
 Text Splitter (RecursiveCharacterTextSplitter)
      │
      ▼
 Embedding Model (sentence-transformers / all-MiniLM-L6-v2)
      │
      ▼
 Chroma Vector Database  ◄──────────────┐
      │                                 │
      ▼                                 │
 Retriever (top-k similarity search)    │  User question
      │                                 │
      ▼                                 │
 Relevant chunks ───────────────────────┘
      │
      ▼
 Local LLM (Ollama, e.g. llama3.1)
      │
      ▼
 Grounded answer + cited source chunks
```

## Tech stack

| Layer            | Tool                                             |
|-------------------|---------------------------------------------------|
| Orchestration      | LangChain                                          |
| PDF parsing        | `langchain_community.document_loaders.PyPDFLoader` |
| Embeddings          | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database     | ChromaDB (persisted locally)                       |
| LLM                | Ollama (local, e.g. `llama3.1`)                    |
| UI                 | Streamlit                                          |

## Project structure

```
ContextIQ/
├── app.py              # Streamlit UI — upload, chat, view sources
├── rag_engine.py        # Core RAG pipeline: load, split, embed, retrieve, generate
├── config.py            # All tunable settings in one place
├── requirements.txt      # Python dependencies
├── data/                 # Uploaded PDFs land here
└── chroma_db/            # Persisted vector index (auto-created)
```

## Setup

### 1. Install Ollama and pull a model

ContextIQ uses [Ollama](https://ollama.com) to run the LLM locally.

```bash
# Install Ollama (see https://ollama.com/download for your OS)
ollama pull llama3.1
ollama serve   # starts the local Ollama server on localhost:11434
```

You can swap in a smaller/faster model (e.g. `llama3.2:3b` or `phi3`) by
editing `OLLAMA_MODEL` in `config.py`.

### 2. Install Python dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

This opens ContextIQ in your browser at `http://localhost:8501`.

## Usage

1. In the sidebar, upload a PDF and click **Ingest PDF**. This splits the
   document into chunks, embeds them, and stores them in a local Chroma
   index (persisted in `chroma_db/`, so it survives app restarts).
2. Ask a question in the chat box. ContextIQ retrieves the most relevant
   chunks and asks the local LLM to answer using *only* that content.
3. Expand **"View retrieved sources"** under any answer to see exactly
   which chunks (and which page) the answer was built from.
4. Use **Clear index** in the sidebar to wipe everything and start fresh,
   or check **Replace existing index** before ingesting a new PDF to swap
   documents instead of accumulating them.

## How it works (the RAG pipeline)

1. **Load** — `PyPDFLoader` extracts text from each page of the PDF.
2. **Split** — `RecursiveCharacterTextSplitter` breaks pages into ~1000
   character chunks with 150 characters of overlap, so context isn't lost
   at chunk boundaries.
3. **Embed** — each chunk is converted into a vector with a local
   Hugging Face sentence-transformer model (no API calls).
4. **Store** — vectors + text are persisted in a Chroma collection on disk.
5. **Retrieve** — when you ask a question, it's embedded the same way and
   compared against stored vectors; the top-k most similar chunks are pulled.
6. **Generate** — those chunks are inserted into a prompt template and sent
   to the local Ollama LLM, which is instructed to answer *only* from that
   context and to say so if the answer isn't there.

## Extending this project

This is designed as a solid base for a portfolio project. Natural upgrades:

- **Multi-document RAG** — index many PDFs at once and filter by source in
  the retriever's metadata.
- **Conversational memory** — chain in chat history so follow-up questions
  ("what about the second one?") resolve correctly.
- **Hybrid search** — combine keyword (BM25) and vector search for better
  recall on exact terms like names or numbers.
- **Re-ranking** — add a cross-encoder re-ranker on top of the initial
  retrieval to improve precision.
- **Swap the LLM** — point `get_llm()` in `rag_engine.py` at a hosted model
  (OpenAI, Anthropic, etc.) instead of Ollama for higher quality answers.
- **Other file types** — add loaders for `.docx`, `.txt`, or web pages
  alongside the PDF loader.

## Troubleshooting

- **"Couldn't reach the local LLM"** — make sure `ollama serve` is running
  and you've pulled the model set in `config.py` (`ollama pull llama3.1`).
- **Slow first ingestion** — the embedding model downloads on first use
  (a few hundred MB); subsequent runs are fast since it's cached locally.
- **Answers seem off-topic** — try lowering `CHUNK_SIZE` for denser PDFs,
  or increasing `TOP_K` in `config.py` so more context is retrieved.
