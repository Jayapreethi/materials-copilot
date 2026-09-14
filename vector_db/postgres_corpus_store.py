"""Relational PostgreSQL + pgvector store for parsed CO2M PDF chunks."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id BIGSERIAL PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    title TEXT,
    authors TEXT,
    publication_year INTEGER,
    page_count INTEGER NOT NULL DEFAULT 0,
    source_dataset TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_sha256 CHAR(64) NOT NULL,
    is_unique BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_unique BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(document_id, chunk_number)
);

CREATE INDEX IF NOT EXISTS documents_year_idx ON documents(publication_year);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS documents_is_unique_idx ON documents(is_unique);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN(metadata);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;
"""


class PostgresCorpusStore:
    """Postgres store with relational document/chunk columns and JSONB metadata."""

    def __init__(self, dsn: Optional[str] = None, dimension: int = 384, schema: str = "public") -> None:
        self.dsn = dsn or os.getenv(
            "CO2M_POSTGRES_DSN",
            "postgresql://co2m:co2m@127.0.0.1:5432/co2m",
        )
        self.dimension = dimension
        if not schema.replace("_", "").isalnum() or not schema:
            raise ValueError("schema must contain only letters, digits, and underscores")
        self.schema = schema
        if dimension != 384:
            raise ValueError("The current schema is defined for 384-dimensional embeddings")
        self.initialize()

    def connect(self):
        return psycopg.connect(
            self.dsn,
            row_factory=dict_row,
            options=f"-c search_path={self.schema},public",
        )

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(SCHEMA_SQL)

    def upsert_document(self, document: dict[str, Any]) -> int:
        with self.connect() as connection:
            row = connection.execute(
                """
                INSERT INTO documents
                    (source_path, filename, title, authors, publication_year,
                     page_count, source_dataset, metadata, content_sha256)
                VALUES (%(source_path)s, %(filename)s, %(title)s, %(authors)s,
                        %(publication_year)s, %(page_count)s, %(source_dataset)s,
                        %(metadata)s::jsonb, %(content_sha256)s)
                ON CONFLICT (source_path) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    publication_year = EXCLUDED.publication_year,
                    page_count = EXCLUDED.page_count,
                    source_dataset = EXCLUDED.source_dataset,
                    metadata = EXCLUDED.metadata,
                    content_sha256 = EXCLUDED.content_sha256,
                    updated_at = now()
                RETURNING document_id
                """,
                {**document, "metadata": json.dumps(document.get("metadata", {}))},
            ).fetchone()
            return int(row["document_id"])

    def replace_document_chunks(self, document_id: int, chunks: Iterable[dict[str, Any]]) -> int:
        rows = list(chunks)
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            for chunk in rows:
                connection.execute(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_number, page_start, page_end, content,
                         metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    (
                        document_id,
                        chunk["chunk_number"],
                        chunk.get("page_start"),
                        chunk.get("page_end"),
                        chunk["content"],
                        json.dumps(chunk.get("metadata", {})),
                        self.vector_literal(chunk["embedding"]) if chunk.get("embedding") else None,
                    ),
                )
        return len(rows)

    @staticmethod
    def vector_literal(values: list[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in values) + "]"

    def search(
        self, embedding: list[float], top_k: int = 5, unique_only: bool = False
    ) -> list[dict[str, Any]]:
        vector = self.vector_literal(embedding)
        unique_clause = "AND d.is_unique = true" if unique_only else ""
        with self.connect() as connection:
            return connection.execute(
                f"""
                SELECT c.chunk_id, c.document_id, c.content, c.page_start, c.page_end,
                       c.metadata AS chunk_metadata, d.filename, d.title,
                       d.publication_year, d.metadata AS document_metadata,
                       c.embedding <=> %s::vector AS distance
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE c.embedding IS NOT NULL {unique_clause}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, vector, top_k),
            ).fetchall()

    def stats(self) -> dict[str, int]:
        with self.connect() as connection:
            documents = connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            embedded = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()["n"]
            unique_documents = connection.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE is_unique = true"
            ).fetchone()["n"]
        return {
            "documents": documents,
            "chunks": chunks,
            "embedded_chunks": embedded,
            "unique_documents": unique_documents,
        }

    def mark_unique_documents(self, filenames: set[str]) -> dict[str, int]:
        """Flip is_unique=true for documents whose filename is in the given set (non-destructive)."""
        with self.connect() as connection:
            connection.execute("UPDATE documents SET is_unique = false WHERE is_unique = true")
            matched = connection.execute(
                "UPDATE documents SET is_unique = true WHERE filename = ANY(%s) RETURNING filename",
                (list(filenames),),
            ).fetchall()
        matched_filenames = {row["filename"] for row in matched}
        return {
            "list_size": len(filenames),
            "matched": len(matched_filenames),
            "missing": len(filenames - matched_filenames),
        }