"""
Download PDFs from PatronusAI/financebench and save them to data/pdfs/.
Run from the repo root:  python data/data.py
"""

import os
import requests
from datasets import load_dataset

PDF_DIR = os.path.join(os.path.dirname(__file__), "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)


def main():
    print("Loading financebench dataset...")
    ds = load_dataset("PatronusAI/financebench", split="train")

    print(f"\nDataset size: {len(ds)} rows")
    print(f"Columns: {ds.column_names}\n")

    # Show a sample row so we can see the structure
    sample = ds[0]
    for key, val in sample.items():
        preview = str(val)[:120]
        print(f"  {key}: {preview}")

    print("\n--- Downloading PDFs ---")

    # Collect unique doc links to avoid downloading the same PDF twice
    seen = set()
    downloaded = 0
    skipped = 0

    for row in ds:
        doc_name = row.get("doc_name", "")
        doc_link = row.get("doc_link", "") or row.get("pdf_link", "")

        if not doc_link or doc_link in seen:
            skipped += 1
            continue
        seen.add(doc_link)

        filename = f"{doc_name}.pdf" if doc_name else os.path.basename(doc_link)
        dest = os.path.join(PDF_DIR, filename)

        if os.path.exists(dest):
            print(f"  [skip] {filename} already exists")
            skipped += 1
            continue

        try:
            resp = requests.get(doc_link, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"  [ok]   {filename}")
            downloaded += 1
        except Exception as e:
            print(f"  [err]  {filename}: {e}")

    print(f"\nDone. Downloaded: {downloaded}, Skipped/existing: {skipped}")
    print(f"PDFs saved to: {PDF_DIR}")


if __name__ == "__main__":
    main()
