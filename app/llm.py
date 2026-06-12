import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information."""


def _get_client() -> OpenAI:
    # Ollama is OpenAI-compatible — same client, different base_url, no real API key needed
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def ask(question: str, context_chunks: list[dict]) -> str:
    """Send question + retrieved chunks to llama3 via Ollama and return the answer."""
    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}]\n{c['text']}"
        for c in context_chunks
    )

    user_message = f"""Context:
{context}

Question: {question}"""

    client = _get_client()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content
