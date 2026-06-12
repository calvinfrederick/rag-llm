import os
import hashlib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between adjacent chunks
COLLECTION_NAME = "documents"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection

# reads every page of the PDF and joins the text into a single string
def _extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# splits text into 500-character chunks with 50-character overlap (the overlap stops a sentence from being cut in half at a chunk boundary)
def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]

# orchestrates the ingestion process
def ingest_pdf(pdf_path: str) -> int:
    """Ingest a PDF into ChromaDB. Returns the number of chunks stored."""
    text = _extract_text(pdf_path)
    chunks = _chunk_text(text)

    embedder = _get_embedder()
    collection = _get_collection()

    embeddings = embedder.encode(chunks).tolist()
    filename = os.path.basename(pdf_path)

    ids = [hashlib.md5(f"{filename}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)
