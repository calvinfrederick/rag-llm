# RAG LLM API

A Retrieval-Augmented Generation (RAG) API that ingests PDF files and answers questions grounded in their content. Uses ChromaDB for vector storage, `sentence-transformers` for local embeddings, and DeepSeek for generation.

## Architecture

```
app/
├── main.py       # FastAPI app — POST /ingest, POST /query, GET /health
├── ingest.py     # PDF → text → chunks → embeddings → ChromaDB
├── retrieval.py  # Embed question → query ChromaDB → return top-k chunks
└── llm.py        # Pass chunks + question to DeepSeek → return answer
```

**Data flow:**

1. `POST /ingest` — PDF uploaded → text extracted → split into 500-char chunks (50-char overlap) → embedded with `all-MiniLM-L6-v2` → upserted into ChromaDB.
2. `POST /query` — question embedded → top-5 chunks retrieved from ChromaDB → chunks + question sent to DeepSeek → answer returned with source citations.

## Setup

**Prerequisites:** Python 3.10+

```bash
# Clone and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Running
```bash
uvicorn app.main:app --reload
```

Interactive API docs: `http://localhost:8000/docs`

## API

### `POST /ingest`

Upload a PDF file to ingest it into the vector store.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{ "filename": "document.pdf", "chunks_stored": 42 }
```

### `POST /query`

Ask a question answered from ingested documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Response:
```json
{
  "answer": "...",
  "sources": [
    { "text": "...", "source": "document.pdf", "chunk": 3, "distance": 0.1234 }
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API URL |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | Model used for generation |
| `DEEPSEEK_API_KEY` | — | Your DeepSeek API key (required) |
