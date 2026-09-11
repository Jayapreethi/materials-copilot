#!/usr/bin/env python
"""Build dashboard parquet artifacts from Sloan data without importing pandas."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(sys.maxsize)


def clean(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def concept(value: object) -> str:
    return (clean(value).replace("_", " ").replace("-", " ").title() or "Unclassified")


def load_records(data_dir: Path, limit: int = 0) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(data_dir.rglob("downloaded_articles_metadata.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("pdf_download_status") != "success":
                    continue
                title = clean(row.get("title"))
                abstract = clean(row.get("abstract"))
                if not abstract:
                    continue
                key = (clean(row.get("dedup_key")), title)
                if key in seen:
                    continue
                seen.add(key)
                records.append(row)
                if limit and len(records) >= limit:
                    return records
    return records


def table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def build(records: list[dict[str, str]], output: Path, chunk_words: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    chunks: list[dict] = []
    concepts: list[dict] = []
    chunk_id = 1

    for doc_id, row in enumerate(records, 1):
        filename = Path(clean(row.get("pdf_local_path")).replace("\\", "/")).name
        filename = filename or f"record_{doc_id:05d}.pdf"
        year = None
        try:
            year = int(row.get("publication_year", ""))
        except (TypeError, ValueError):
            pass
        documents.append({
            "doc_id": doc_id,
            "filename": filename,
            "title": clean(row.get("title")) or filename,
            "authors": clean(row.get("authors")),
            "year": year,
            "page_count": None,
        })
        name = concept(row.get("technology_sub_domain"))
        words = clean(row.get("abstract")).split()
        for start in range(0, len(words), chunk_words):
            text = " ".join(words[start:start + chunk_words])
            if not text:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_text": text,
                "page_number": 1,
            })
            concepts.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "concept_name": name,
                "confidence_score": 1.0,
            })
            chunk_id += 1

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"doc_count": 0, "chunk_count": 0})
    doc_seen: dict[str, set[int]] = defaultdict(set)
    for item in concepts:
        doc_seen[item["concept_name"]].add(item["doc_id"])
        counts[item["concept_name"]]["chunk_count"] += 1
    concept_counts = [
        {"concept_name": name, "doc_count": len(doc_seen[name]), "chunk_count": values["chunk_count"]}
        for name, values in sorted(counts.items())
    ]
    summaries = [
        {
            "concept_name": row["concept_name"],
            "summary_text": f"{row['doc_count']:,} documents and {row['chunk_count']:,} abstract chunks are classified under {row['concept_name'].lower()}.",
        }
        for row in concept_counts
    ]

    by_concept_doc: dict[tuple[str, int], int] = Counter(
        (item["concept_name"], item["doc_id"]) for item in concepts
    )
    doc_lookup = {item["doc_id"]: item for item in documents}
    top_docs = []
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (name, doc_id), count in by_concept_doc.items():
        grouped[name].append((doc_id, count))
    for name, values in grouped.items():
        for rank, (doc_id, count) in enumerate(sorted(values, key=lambda pair: -pair[1]), 1):
            top_docs.append({
                "concept_name": name,
                "doc_id": doc_id,
                "filename": doc_lookup[doc_id]["filename"],
                "title": doc_lookup[doc_id]["title"],
                "chunk_count": count,
                "rank": rank,
            })

    text_by_concept: dict[str, list[str]] = defaultdict(list)
    chunk_lookup = {item["chunk_id"]: item["chunk_text"] for item in chunks}
    for item in concepts:
        text_by_concept[item["concept_name"]].append(chunk_lookup[item["chunk_id"]])
    keywords = []
    for name, texts in text_by_concept.items():
        word_counts = Counter(
            word for text in texts for word in re.findall(r"[A-Za-z]{4,}", text.lower())
        )
        total = max(sum(word_counts.values()), 1)
        for word, count in word_counts.most_common(50):
            keywords.append({"concept_name": name, "keyword": word, "weight": count / total})

    frames = {
        "documents": documents,
        "chunks": chunks,
        "concepts": concepts,
        "summaries": summaries,
        "concept_counts": concept_counts,
        "top_documents_by_concept": top_docs,
        "keywords_by_concept": keywords,
    }
    for name, rows in frames.items():
        pq.write_table(table(rows), output / f"{name}.parquet")

    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "data folder metadata abstracts",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "concept_assignment_count": len(concepts),
        "source_files": len(records),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("co2m/artifacts"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-words", type=int, default=180)
    args = parser.parse_args()
    print("Scanning metadata files...", flush=True)
    records = load_records(args.data_dir, args.limit)
    print(f"Loaded {len(records):,} records; writing artifacts...", flush=True)
    print(json.dumps(build(records, args.out, args.chunk_words), indent=2))


if __name__ == "__main__":
    main()
