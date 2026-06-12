# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG (Retrieval-Augmented Generation) API that ingests PDF files and answers questions grounded in their content. Uses ChromaDB for vector storage, `sentence-transformers` for local embeddings, and llama3 via Ollama for generation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Requires Ollama running locally with llama3:
```bash
ollama pull llama3
ollama serve
```

## Running the server

```bash
uvicorn app.main:app --reload
```

Interactive API docs available at `http://localhost:8000/docs`.

## Key commands

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

# Health check
curl http://localhost:8000/health
```

## Architecture

```
app/
├── main.py       # FastAPI app — POST /ingest, POST /query, GET /health
├── ingest.py     # PDF → text → chunks → embeddings → ChromaDB
├── retrieval.py  # Embed question → query ChromaDB → return top-k chunks
└── llm.py        # Pass chunks + question to llama3 via Ollama → return answer
```

**Data flow:**

1. `POST /ingest` — PDF uploaded → `ingest.py` extracts text, splits into 500-char chunks (50-char overlap), embeds with `all-MiniLM-L6-v2`, upserts into ChromaDB.
2. `POST /query` — question embedded → `retrieval.py` queries ChromaDB for top-5 chunks → `llm.py` builds prompt with context and calls llama3 → answer returned with source citations.

## Key constants (ingest.py)

- `CHUNK_SIZE = 500` — characters per chunk
- `CHUNK_OVERLAP = 50` — overlap between adjacent chunks
- `TOP_K = 5` — number of chunks retrieved per query (retrieval.py)

## Environment variables (.env)

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Model to use for generation |

## Notes

- `chroma_db/` is gitignored — vector index is local only, re-ingest after cloning.
- Re-ingesting the same PDF is safe — chunks are upserted by a deterministic ID (MD5 of filename + chunk index).
- To switch to OpenAI GPT: set `OLLAMA_BASE_URL=https://api.openai.com/v1`, `OLLAMA_MODEL=gpt-4o-mini`, and add `OPENAI_API_KEY` — the same `llm.py` client works unchanged.
