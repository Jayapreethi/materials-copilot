"""PostgreSQL/pgvector backend compatible with the existing VectorStore API."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row


class PostgresVectorStore:
    """Store and search embeddings in PostgreSQL using the pgvector extension."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        collection_name: str = "co2m_corpus",
        dimension: int = 384,
        distance_metric: str = "cosine",
    ) -> None:
        if distance_metric != "cosine":
            raise ValueError("PostgresVectorStore currently supports cosine distance only")
        self.dsn = dsn or os.getenv(
            "CO2M_POSTGRES_DSN",
            "postgresql://co2m:co2m@127.0.0.1:5432/co2m",
        )
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance_metric = distance_metric
        self.table_name = self._safe_identifier(collection_name)
        self._initialize_schema()

    @staticmethod
    def _safe_identifier(value: str) -> str:
        if not value.replace("_", "").isalnum() or not value[0].isalpha():
            raise ValueError(f"Invalid collection name: {value!r}")
        return f"co2m_vectors_{value}"

    @staticmethod
    def _vector_literal(values: List[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in values) + "]"

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        document TEXT NOT NULL,
                        embedding vector({self.dimension}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name} USING hnsw (embedding vector_cosine_ops)
                """)

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        if not (len(texts) == len(embeddings) == len(metadatas) == len(ids)):
            raise ValueError("texts, embeddings, metadatas, and ids must have equal lengths")
        if not texts:
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_name} (id, document, embedding, metadata)
                        VALUES (%s, %s, %s::vector, %s::jsonb)
                        ON CONFLICT (id) DO UPDATE SET
                            document = EXCLUDED.document,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        (doc_id, text, self._vector_literal(embedding), json.dumps(metadata)),
                    )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        conditions: List[str] = []
        parameters: List[Any] = [self._vector_literal(query_embedding)]
        if where:
            for key, condition in where.items():
                if not isinstance(condition, dict) or set(condition) != {"$eq"}:
                    raise ValueError("Only {$eq: value} metadata filters are supported")
                conditions.append("metadata ->> %s = %s")
                parameters.extend([key, str(condition["$eq"])])
        filter_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(top_k)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, document, metadata,
                           embedding <=> %s::vector AS distance
                    FROM {self.table_name}
                    {filter_sql}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [parameters[0], *parameters[1:-1], parameters[0], parameters[-1]],
                )
                rows = cursor.fetchall()
        return {
            "ids": [row["id"] for row in rows],
            "distances": [float(row["distance"]) for row in rows],
            "metadatas": [row["metadata"] for row in rows],
            "documents": [row["document"] for row in rows],
        }

    def delete_by_filename(self, filename: str) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE metadata ->> 'filename' = %s",
                    (filename,),
                )
                return cursor.rowcount

    def persist(self) -> None:
        """PostgreSQL commits each operation; no explicit persist is needed."""

    def get_stats(self) -> Dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) AS count FROM {self.table_name}")
                count = cursor.fetchone()["count"]
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "distance_metric": self.distance_metric,
            "backend": "postgresql+pgvector",
            "table": self.table_name,
        }

    @staticmethod
    def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as file_handle:
            for byte_block in iter(lambda: file_handle.read(chunk_size), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()