from app.ingest import _get_embedder, _get_collection

TOP_K = 5

def retrieve(question: str) -> list[dict]:
    """Return the top-k most relevant chunks for a given question.

    Each result has:
      - text     : the chunk content
      - source   : filename the chunk came from
      - chunk    : chunk index within that file
      - distance : lower = more similar
    """
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", -1),
            "distance": round(distance, 4),
        })

    return chunks
