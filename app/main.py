import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.responses import StreamingResponse
from pydantic import BaseModel

from app.ingest import ingest_pdf
from app.retrieval import retrieve
from app.llm import ask, ask_stream

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        num_chunks = ingest_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {"filename": file.filename, "chunks_stored": num_chunks}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question and get an answer grounded in ingested documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(request.question)

    if not chunks:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")

    answer = ask(request.question, chunks)

    return QueryResponse(answer=answer, sources=chunks)


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Ask a question and stream the answer token-by-token."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(request.question)

    if not chunks:
        raise HTTPException(status_code=404, detail="No documents ingested yet.")

    return StreamingResponse(ask_stream(request.question, chunks), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok"}
