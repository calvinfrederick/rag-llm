import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information."""


_client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY)


def _build_messages(question: str, context_chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk']}]\n{c['text']}"
        for c in context_chunks
    )
    user_message = f"""Context:
{context}

Question: {question}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def ask(question: str, context_chunks: list[dict]) -> str:
    """Send question + retrieved chunks to llama3 via Ollama and return the answer."""
    response = _client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=_build_messages(question, context_chunks),
    )
    return response.choices[0].message.content


def ask_stream(question: str, context_chunks: list[dict]):
    """Like ask(), but yields text tokens as they're generated."""
    for chunk in _client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=_build_messages(question, context_chunks),
        stream=True,
    ):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
