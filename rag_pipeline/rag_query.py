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
from local_llm import LocalTransformersModel
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


def transformers_chat(local_model: LocalTransformersModel, question: str, context: str) -> str:
    return local_model.generate_text(
        SYSTEM_PROMPT,
        f"Sources:\n{context}\n\nQuestion: {question}",
        max_new_tokens=2048,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--schema", default="public", help="PostgreSQL schema containing the corpus")
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=None, help="Ollama model name or local Hugging Face model directory")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow Transformers to download a model if it is not cached locally")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-generate", action="store_true", help="Only print retrieved chunks, skip LLM call")
    parser.add_argument("--all-documents", action="store_true", help="Include duplicate PDFs, not just the unique set")
    args = parser.parse_args()
    args.model = args.model or ("qwen3:8b" if args.backend == "ollama" else None)
    if args.backend == "transformers" and not args.model:
        parser.error("--model must be a local Hugging Face model directory for --backend transformers")

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
    if args.backend == "ollama":
        answer = ollama_chat(args.ollama_url, args.model, args.question, context, args.timeout)
    else:
        local_model = LocalTransformersModel(
            args.model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            local_files_only=not args.allow_model_download,
        )
        answer = transformers_chat(local_model, args.question, context)
    print(answer)


if __name__ == "__main__":
    main()
