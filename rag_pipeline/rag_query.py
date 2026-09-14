#!/usr/bin/env python
"""
RAG query CLI: retrieves full-PDF chunks from Postgres/pgvector (built by
ingest_postgres.py) and generates a grounded answer via a local Ollama model.

Usage:
    python rag_pipeline/rag_query.py "What CO2 uptake do MOFs achieve?"
    python rag_pipeline/rag_query.py --top-k 8 --model qwen3:8b "..."
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embeddings import EmbeddingGenerator
from vector_db.postgres_corpus_store import PostgresCorpusStore

SYSTEM_PROMPT = (
    "You are a scientific literature assistant. Answer the question using ONLY "
    "the numbered source excerpts provided. Cite sources inline as [1], [2], etc. "
    "If the excerpts do not contain enough information, say so explicitly."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, 1):
        pages = f"p.{chunk['page_start']}" if chunk.get("page_start") else ""
        parts.append(
            f"[{idx}] ({chunk.get('title') or chunk.get('filename')}, {pages})\n{chunk['content']}"
        )
    return "\n\n".join(parts)


def ollama_chat(base_url: str, model: str, question: str, context: str, timeout: int) -> str:
    body = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--schema", default="public", help="PostgreSQL schema containing the corpus")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-generate", action="store_true", help="Only print retrieved chunks, skip LLM call")
    parser.add_argument("--all-documents", action="store_true", help="Include duplicate PDFs, not just the unique set")
    args = parser.parse_args()

    embedder = EmbeddingGenerator()
    store = PostgresCorpusStore(dsn=args.dsn, schema=args.schema)

    query_embedding = embedder.embed([args.question])[0].tolist()
    chunks = store.search(query_embedding, top_k=args.top_k, unique_only=not args.all_documents)

    if not chunks:
        print("No chunks found. Has ingestion completed?")
        return

    context = build_context(chunks)
    print("Retrieved sources:")
    for idx, chunk in enumerate(chunks, 1):
        print(f"  [{idx}] {chunk.get('filename')} p.{chunk.get('page_start')} "
              f"(distance={chunk['distance']:.3f})")

    if args.no_generate:
        return

    print("\nGenerating answer...\n")
    answer = ollama_chat(args.ollama_url, args.model, args.question, context, args.timeout)
    print(answer)


if __name__ == "__main__":
    main()
