#!/usr/bin/env python
"""Parse Sloan PDFs, create chunks/embeddings, and load PostgreSQL + pgvector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

def clean(value: object) -> str:
    text = html.unescape(str(value or "")).replace("\x00", "")
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def configure_csv_field_limit() -> None:
    """Allow metadata fields such as raw_source_metadata to exceed 128 KiB."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def metadata_index(metadata_dirs: list[Path]) -> dict[str, dict[str, str]]:
    configure_csv_field_limit()
    index: dict[str, dict[str, str]] = {}
    for metadata_dir in metadata_dirs:
        csv_paths = metadata_dir.rglob("downloaded_articles_metadata.csv")
        for csv_path in csv_paths:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    pdf_path = Path(clean(row.get("pdf_local_path"))).name
                    if pdf_path:
                        index.setdefault(pdf_path.lower(), row)
        for jsonl_path in metadata_dir.rglob("qwen_metadata.jsonl"):
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    filename = clean(record.get("filename")).lower()
                    if filename:
                        existing = index.setdefault(filename, {})
                        existing["llm_metadata"] = record.get("metadata", {})
                        existing["llm_model"] = record.get("model")
                        existing["llm_source_pages"] = record.get("source_pages", [])
    return index


def sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {clean(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return clean(value)
    return value


def parse_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    pages = []
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, 1):
        text = clean(page.extract_text())
        if text:
            pages.append((page_number, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], words_per_chunk: int) -> list[dict]:
    chunks: list[dict] = []
    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), words_per_chunk):
            content = " ".join(words[start:start + words_per_chunk]).strip()
            if content:
                chunks.append({
                    "chunk_number": len(chunks),
                    "page_start": page_number,
                    "page_end": page_number,
                    "content": content,
                    "metadata": {"parser": "pypdf", "word_start": start},
                })
    return chunks


def process(pdf_path: Path, data_root: Path, metadata: dict[str, dict[str, str]], store, embedder, words_per_chunk: int) -> int:
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages, words_per_chunk)
    if not chunks:
        return 0
    vectors = embedder.embed([chunk["content"] for chunk in chunks]).tolist()
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    relative = str(pdf_path.relative_to(data_root)).replace("\\", "/")
    row = metadata.get(pdf_path.name.lower(), {})
    llm_metadata = row.get("llm_metadata", {})
    year = None
    try:
        year = int(row.get("publication_year") or llm_metadata.get("publication_year"))
    except (TypeError, ValueError):
        pass
    llm_authors = llm_metadata.get("authors", [])
    authors = clean(row.get("authors")) or ", ".join(clean(author) for author in llm_authors)
    document = {
        "source_path": relative,
        "filename": pdf_path.name,
        "title": clean(row.get("title")) or clean(llm_metadata.get("title")) or pdf_path.stem,
        "authors": authors,
        "publication_year": year,
        "page_count": len(pages),
        "source_dataset": pdf_path.parts[0] if pdf_path.parts else None,
        "metadata": {key: sanitize_json(value) for key, value in row.items() if value},
        "content_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    }
    document_id = store.upsert_document(document)
    return store.replace_document_chunks(document_id, chunks)


def main() -> None:
    from embeddings.embeddings import EmbeddingGenerator
    from vector_db.postgres_corpus_store import PostgresCorpusStore

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, action="append", required=True,
                        help="Directory containing PDFs; repeat for multiple datasets")
    parser.add_argument("--metadata-dir", type=Path, action="append", default=[],
                        help="Directory containing downloaded_articles_metadata.csv")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pdfs = sorted({pdf for data_dir in args.data_dir for pdf in data_dir.rglob("*.pdf")})
    if args.limit:
        pdfs = pdfs[:args.limit]
    metadata_dirs = args.metadata_dir or args.data_dir
    metadata = metadata_index(metadata_dirs)
    store = PostgresCorpusStore(dsn=args.dsn)
    embedder = EmbeddingGenerator(device=args.device)
    total_chunks = 0
    for number, pdf_path in enumerate(pdfs, 1):
        data_root = Path(args.data_dir[0]).parent if len(args.data_dir) > 1 else args.data_dir[0]
        try:
            count = process(pdf_path, data_root, metadata, store, embedder, args.chunk_words)
        except Exception as exc:
            print(f"skipped={pdf_path.name} error={exc}", flush=True)
            continue
        total_chunks += count
        if number % 25 == 0:
            print(f"processed={number}/{len(pdfs)} chunks={total_chunks}", flush=True)
    print(store.stats())


if __name__ == "__main__":
    main()