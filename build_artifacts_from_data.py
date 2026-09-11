#!/usr/bin/env python
"""Build dashboard artifacts from the Sloan data-folder metadata.

The data folder contains one metadata CSV per topic output. Abstracts provide
stable source text when a PDF text-extraction dependency is unavailable.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED_OUTPUTS = [
    "documents.parquet",
    "chunks.parquet",
    "concepts.parquet",
    "summaries.parquet",
    "concept_counts.parquet",
    "top_documents_by_concept.parquet",
    "keywords_by_concept.parquet",
]


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def concept_label(value: object) -> str:
    text = clean_text(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() or "Unclassified"


def load_records(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metadata_path in sorted(data_dir.rglob("downloaded_articles_metadata.csv")):
        frame = pd.read_csv(metadata_path, low_memory=False)
        frame["_source_dir"] = str(metadata_path.parent.parent)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No downloaded_articles_metadata.csv files found under {data_dir}")

    records = pd.concat(frames, ignore_index=True)
    if "pdf_download_status" in records:
        records = records[records["pdf_download_status"].eq("success")].copy()
    records["title"] = records.get("title", "").map(clean_text)
    records["abstract"] = records.get("abstract", "").map(clean_text)
    records = records[records["abstract"].str.len().gt(0)].copy()
    records["dedup_key"] = records.get("dedup_key", records["title"])
    records = records.drop_duplicates(subset=["dedup_key", "title"], keep="first")
    return records.reset_index(drop=True)


def build_artifacts(records: pd.DataFrame, output_dir: Path, chunk_words: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    chunks: list[dict] = []
    concepts: list[dict] = []
    doc_concepts: dict[int, set[str]] = {}
    chunk_id = 1

    for doc_id, (_, record) in enumerate(records.iterrows(), start=1):
        filename = clean_text(record.get("pdf_local_path", ""))
        filename = Path(filename.replace("\\", "/")).name or f"record_{doc_id:05d}.pdf"
        try:
            year = int(record.get("publication_year"))
        except (TypeError, ValueError):
            year = None
        title = record.get("title") or filename
        documents.append({
            "doc_id": doc_id,
            "filename": filename,
            "title": title,
            "authors": clean_text(record.get("authors", "")),
            "year": year,
            "page_count": None,
        })

        concept = concept_label(record.get("technology_sub_domain"))
        doc_concepts[doc_id] = {concept}
        words = record["abstract"].split()
        for start in range(0, len(words), chunk_words):
            text = " ".join(words[start:start + chunk_words]).strip()
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
                "concept_name": concept,
                "confidence_score": 1.0,
            })
            chunk_id += 1

    documents_df = pd.DataFrame(documents)
    chunks_df = pd.DataFrame(chunks)
    concepts_df = pd.DataFrame(concepts)
    concept_counts_df = (
        concepts_df.groupby("concept_name")
        .agg(doc_count=("doc_id", "nunique"), chunk_count=("chunk_id", "nunique"))
        .reset_index()
    )
    summaries_df = pd.DataFrame([
        {
            "concept_name": concept,
            "summary_text": (
                f"{int(row.doc_count):,} documents and {int(row.chunk_count):,} abstract chunks "
                f"are classified under {concept.lower()}."
            ),
        }
        for row in concept_counts_df.itertuples(index=False)
        for concept in [row.concept_name]
    ])
    top_docs_df = (
        concepts_df.groupby(["concept_name", "doc_id"])
        .size().reset_index(name="chunk_count")
        .merge(documents_df[["doc_id", "filename", "title"]], on="doc_id", how="left")
        .sort_values(["concept_name", "chunk_count"], ascending=[True, False])
    )
    top_docs_df["rank"] = top_docs_df.groupby("concept_name").cumcount() + 1

    keyword_rows: list[dict] = []
    for concept, group in concepts_df.groupby("concept_name"):
        texts = chunks_df[chunks_df["chunk_id"].isin(group["chunk_id"])]
        counts: dict[str, int] = {}
        for text in texts["chunk_text"]:
            for word in re.findall(r"[A-Za-z]{4,}", text.lower()):
                counts[word] = counts.get(word, 0) + 1
        total = max(sum(counts.values()), 1)
        for word, count in sorted(counts.items(), key=lambda pair: -pair[1])[:50]:
            keyword_rows.append({
                "concept_name": concept,
                "keyword": word,
                "weight": count / total,
            })
    keywords_df = pd.DataFrame(keyword_rows, columns=["concept_name", "keyword", "weight"])

    dataframes = {
        "documents": documents_df,
        "chunks": chunks_df,
        "concepts": concepts_df,
        "summaries": summaries_df,
        "concept_counts": concept_counts_df,
        "top_documents_by_concept": top_docs_df,
        "keywords_by_concept": keywords_df,
    }
    for name, frame in dataframes.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "data folder metadata abstracts",
        "document_count": len(documents_df),
        "chunk_count": len(chunks_df),
        "concept_assignment_count": len(concepts_df),
        "source_files": len(records),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("co2m/artifacts"))
    parser.add_argument("--limit", type=int, default=0, help="Limit records; 0 builds all records")
    parser.add_argument("--chunk-words", type=int, default=180)
    args = parser.parse_args()

    records = load_records(args.data_dir)
    if args.limit > 0:
        records = records.head(args.limit)
    manifest = build_artifacts(records, args.out, args.chunk_words)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
