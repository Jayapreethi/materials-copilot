#!/usr/bin/env python
"""Read-only quality evaluation for the Postgres + pgvector corpus.

Precision and recall require human relevance judgments. Supply them as JSONL:
    {"query": "CO2 capture MOFs", "relevant_chunk_ids": [12, 34]}
or:
    {"query": "CO2 capture MOFs", "relevant_filenames": ["paper.pdf"]}

The structural and self-retrieval tests need no labels and never write data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embeddings import EmbeddingGenerator
from vector_db.postgres_corpus_store import PostgresCorpusStore


def load_qrels(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("query"), str) or not record["query"].strip():
            raise ValueError(f"{path}:{line_number}: query must be a non-empty string")
        if not record.get("relevant_chunk_ids") and not record.get("relevant_filenames"):
            raise ValueError(
                f"{path}:{line_number}: provide relevant_chunk_ids or relevant_filenames"
            )
        records.append(record)
    if not records:
        raise ValueError(f"No judgments found in {path}")
    return records


def structural_checks(store: PostgresCorpusStore, chunk_words: int) -> dict[str, Any]:
    with store.connect() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS chunks,
                   COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings,
                   COUNT(*) FILTER (WHERE btrim(content) = '') AS blank_chunks,
                   COUNT(*) FILTER (WHERE page_start IS NULL OR page_end IS NULL) AS missing_pages,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY cardinality(regexp_split_to_array(btrim(content), '\\s+'))) AS median_words,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY cardinality(regexp_split_to_array(btrim(content), '\\s+'))) AS p95_words
            FROM chunks
            """
        ).fetchone()
        unchunked = connection.execute(
            """
            SELECT COUNT(*) AS count FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.document_id)
            """
        ).fetchone()["count"]
        dimensions = connection.execute(
            """
            SELECT vector_dims(embedding) AS dimensions, COUNT(*) AS count
            FROM chunks WHERE embedding IS NOT NULL GROUP BY 1 ORDER BY 1
            """
        ).fetchall()

    median_words = float(row["median_words"] or 0)
    p95_words = float(row["p95_words"] or 0)
    return {
        **dict(row),
        "documents_without_chunks": unchunked,
        "embedding_dimensions": [dict(item) for item in dimensions],
        "configured_chunk_words": chunk_words,
        "median_within_target": 0 < median_words <= chunk_words,
        "p95_within_target_plus_one_word": p95_words <= chunk_words + 1,
    }


def self_retrieval_control(store: PostgresCorpusStore, sample_size: int) -> dict[str, Any]:
    with store.connect() as connection:
        samples = connection.execute(
            """
            SELECT chunk_id, embedding FROM chunks
            WHERE embedding IS NOT NULL ORDER BY chunk_id LIMIT %s
            """,
            (sample_size,),
        ).fetchall()
    hits = 0
    for sample in samples:
        embedding = [float(value) for value in str(sample["embedding"])[1:-1].split(",")]
        result = store.search(embedding, top_k=1, unique_only=False)
        hits += bool(result and result[0]["chunk_id"] == sample["chunk_id"])
    tested = len(samples)
    score = hits / tested if tested else 0.0
    return {
        "tested": tested,
        "top_1_hits": hits,
        "precision_at_1": score,
        "recall_at_1": score,
        "mrr": score,
        "ndcg_at_1": score,
    }


def evaluate_qrels(store: PostgresCorpusStore, qrels: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    embedder = EmbeddingGenerator()
    precision_total = recall_total = reciprocal_rank_total = ndcg_total = 0.0
    details = []
    for record, vector in zip(qrels, embedder.embed([item["query"] for item in qrels]).tolist()):
        results = store.search(vector, top_k=top_k, unique_only=False)
        returned_ids = {item["chunk_id"] for item in results}
        returned_filenames = {item["filename"] for item in results}
        relevant_ids = set(record.get("relevant_chunk_ids", []))
        relevant_filenames = set(record.get("relevant_filenames", []))
        relevant_count = len(relevant_ids or relevant_filenames)
        matches = [
            item for item in results
            if item["chunk_id"] in relevant_ids or item["filename"] in relevant_filenames
        ]
        precision = len(matches) / top_k
        recall = len(matches) / relevant_count
        first_rank = next((rank for rank, item in enumerate(results, 1) if item in matches), None)
        reciprocal_rank = 1 / first_rank if first_rank else 0.0
        dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(results, 1) if item in matches)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant_count, top_k) + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        precision_total += precision
        recall_total += recall
        reciprocal_rank_total += reciprocal_rank
        ndcg_total += ndcg
        details.append({"query": record["query"], "precision_at_k": precision, "recall_at_k": recall,
                        "reciprocal_rank": reciprocal_rank, "ndcg_at_k": ndcg,
                        "returned_chunk_ids": sorted(returned_ids), "returned_filenames": sorted(returned_filenames)})
    count = len(qrels)
    return {"queries": count, "k": top_k, "mean_precision_at_k": precision_total / count,
            "mean_recall_at_k": recall_total / count, "mrr": reciprocal_rank_total / count,
            "mean_ndcg_at_k": ndcg_total / count, "per_query": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only pgvector retrieval and chunk-quality evaluation")
    parser.add_argument("--qrels", type=Path, help="JSONL human relevance judgments")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--self-retrieval-samples", type=int, default=25)
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--schema", default="public", help="PostgreSQL schema containing the corpus")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.top_k < 1 or args.self_retrieval_samples < 1:
        parser.error("--top-k and --self-retrieval-samples must be positive")

    store = PostgresCorpusStore(dsn=args.dsn, schema=args.schema)
    report: dict[str, Any] = {
        "structural_checks": structural_checks(store, args.chunk_words),
        "self_retrieval_control": self_retrieval_control(store, args.self_retrieval_samples),
    }
    if args.qrels:
        report["labeled_retrieval_metrics"] = evaluate_qrels(store, load_qrels(args.qrels), args.top_k)
    print(json.dumps(report, indent=2, default=str))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()