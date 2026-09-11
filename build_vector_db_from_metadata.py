#!/usr/bin/env python
"""Build a Chroma vector database from unique metadata JSON records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from embeddings.embeddings import EmbeddingGenerator
from vector_db.vector_store import VectorStore


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(text_value(item) for item in value if text_value(item))
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {text_value(item)}"
            for key, item in value.items()
            if text_value(item)
        )
    return str(value).strip()


def record_text(record: dict) -> str:
    document = record.get("document", {})
    screening = record.get("screening", {})
    study = record.get("study", {})
    process_system = record.get("process_system", {})
    fields = [
        document.get("filename"),
        document.get("title"),
        document.get("authors"),
        document.get("journal_or_source"),
        document.get("doi"),
        screening.get("relevance_evidence"),
        screening.get("inclusion_decision"),
        study.get("study_scope"),
        process_system.get("process_description"),
        process_system.get("unit_operations"),
    ]
    return " | ".join(text_value(field) for field in fields if text_value(field))


def clean_metadata(record: dict) -> dict:
    document = record.get("document", {})
    screening = record.get("screening", {})
    return {
        "filename": text_value(document.get("filename")),
        "document_id": text_value(document.get("document_id")),
        "title": text_value(document.get("title")),
        "publication_year": text_value(document.get("publication_year")),
        "doi": text_value(document.get("doi")),
        "discovery_source": text_value(document.get("discovery_source")),
        "inclusion_decision": text_value(screening.get("inclusion_decision")),
        "primary_topic": text_value(record.get("taxonomy", {}).get("primary_topic")),
    }


def build(input_path: Path, persist_directory: Path, collection_name: str, batch_size: int) -> dict:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("The input metadata file must contain a JSON list")

    records = [record for record in records if isinstance(record, dict)]
    records = [record for record in records if record.get("document", {}).get("filename")]
    texts = [record_text(record) for record in records]
    ids = [
        "pdf_" + hashlib.sha256(record["document"]["filename"].encode("utf-8")).hexdigest()[:32]
        for record in records
    ]
    metadatas = [clean_metadata(record) for record in records]

    embedding_generator = EmbeddingGenerator(batch_size=batch_size)
    store = VectorStore(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )
    store.client.delete_collection(name=collection_name)
    store = VectorStore(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        embeddings = embedding_generator.embed(texts[start:end]).tolist()
        store.add_documents(
            texts=texts[start:end],
            embeddings=embeddings,
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    manifest = {
        "source_metadata": str(input_path),
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "embedding_model": embedding_generator.model_name,
        "source_records": len(records),
        "vector_count": store.collection.count(),
    }
    persist_directory.mkdir(parents=True, exist_ok=True)
    (persist_directory / "metadata_vector_db_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("co2m/metadata/qwen80b_10903_unique_pdfs_metadata.json"),
    )
    parser.add_argument("--persist-directory", type=Path, default=Path("vector_db/unique_metadata_data"))
    parser.add_argument("--collection", default="co2m_unique_metadata")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.persist_directory, args.collection, args.batch_size), indent=2))


if __name__ == "__main__":
    main()