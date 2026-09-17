#!/usr/bin/env python3
"""
Ingest Daraz policy PDFs into a FAISS vector index.

Output:
  faiss_index/
    index.faiss
    metadata.json

Metadata per chunk includes:
  id, department, source_file, chunk_index, text
"""

from pathlib import Path
import json
import re

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Approximate character-based chunking. Overlap preserves context between chunks.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)

        # Prefer a paragraph/sentence boundary near the target end.
        if end < n:
            candidates = [
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]
            best = max(candidates)
            if best > start + int(size * 0.55):
                end = best + (2 if text[best:best+2] == "\n\n" else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def department_for(pdf_path: Path) -> str:
    return pdf_path.parent.name


def read_pdfs():
    pdfs = sorted(ROOT.rglob("*.pdf"))
    records = []

    for pdf_path in pdfs:
        # Never ingest files already inside the generated FAISS directory.
        if INDEX_DIR in pdf_path.parents:
            continue

        department = department_for(pdf_path)

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            print(f"[WARN] Could not open {pdf_path}: {exc}")
            continue

        full_text = "\n\n".join(page.get_text("text") for page in doc)
        doc.close()

        chunks = chunk_text(full_text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{department}:{pdf_path.stem}:{chunk_idx:05d}"
            records.append({
                "id": chunk_id,
                "department": department,
                "source_file": str(pdf_path.relative_to(ROOT)),
                "chunk_index": chunk_idx,
                "text": chunk,
            })

    return records


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    records = read_pdfs()
    if not records:
        raise RuntimeError("No PDF text/chunks found.")

    texts = [r["text"] for r in records]

    print(f"Embedding {len(texts)} chunks with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # Inner product on normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    config = {
        "embedding_model": MODEL_NAME,
        "metric": "cosine_similarity_via_inner_product",
        "dimension": int(embeddings.shape[1]),
        "chunks": len(records),
    }
    with open(INDEX_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Done. Wrote {INDEX_DIR / 'index.faiss'}")
    print(f"Done. Wrote {INDEX_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
