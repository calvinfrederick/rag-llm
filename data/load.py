"""
Ingest all PDFs in data/pdfs/ into ChromaDB via the /ingest endpoint.
Requires the FastAPI server to be running: uvicorn app.main:app --reload

Run from repo root: python data/load.py
"""

import os
import requests

PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")
INGEST_URL = "http://localhost:8000/ingest"


def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Run data/data.py first.")
        return

    print(f"Found {len(pdf_files)} PDFs. Ingesting...\n")

    success, failed = 0, 0

    for filename in sorted(pdf_files):
        path = os.path.join(PDF_DIR, filename)
        try:
            with open(path, "rb") as f:
                resp = requests.post(INGEST_URL, files={"file": (filename, f, "application/pdf")}, timeout=120)
            resp.raise_for_status()
            chunks = resp.json().get("chunks_stored", "?")
            print(f"  [ok]  {filename} → {chunks} chunks")
            success += 1
        except Exception as e:
            print(f"  [err] {filename}: {e}")
            failed += 1

    print(f"\nDone. Ingested: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
