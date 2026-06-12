# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a new RAG (Retrieval-Augmented Generation) + LLM project. The intended stack, inferred from `.gitignore`, is:

- **Python** (with `venv`/`.venv` for isolation)
- **ChromaDB** for vector storage (`chroma_db/` directory, excluded from git)
- **Jupyter notebooks** for experimentation (`.ipynb_checkpoints/` excluded)
- **`.env`** for secrets (API keys — never commit)

## Setup

Once the project has code, the typical setup will be:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys
```

## Notes

- `chroma_db/` and `*.sqlite3` are gitignored — vector indexes are local only.
- All secrets (API keys, tokens) go in `.env`, which is gitignored.
- This file should be updated as the project structure and commands are established.
