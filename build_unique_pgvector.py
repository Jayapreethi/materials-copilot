#!/usr/bin/env python
"""Build an independent pgvector schema containing only unique PDF chunks.

The source public schema is never modified. By default the target schema is
co2m_unique and is rebuilt from documents where public.documents.is_unique is true.
"""

from __future__ import annotations

import argparse
import os

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


def validate_schema_name(value: str) -> str:
    if not value.replace("_", "").isalnum() or not value:
        raise argparse.ArgumentTypeError("schema name must contain only letters, digits, and underscores")
    return value


def count_rows(connection: psycopg.Connection, schema: str) -> dict[str, int]:
    documents = connection.execute(
        sql.SQL("SELECT COUNT(*) AS count FROM {}.documents").format(sql.Identifier(schema))
    ).fetchone()["count"]
    chunks = connection.execute(
        sql.SQL("SELECT COUNT(*) AS count FROM {}.chunks").format(sql.Identifier(schema))
    ).fetchone()["count"]
    embedded = connection.execute(
        sql.SQL("SELECT COUNT(*) AS count FROM {}.chunks WHERE embedding IS NOT NULL").format(
            sql.Identifier(schema)
        )
    ).fetchone()["count"]
    return {"documents": documents, "chunks": chunks, "embedded_chunks": embedded}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone unique-only pgvector corpus schema")
    parser.add_argument("--schema", type=validate_schema_name, default="co2m_unique")
    parser.add_argument("--dsn", default=os.getenv("CO2M_POSTGRES_DSN", "postgresql://co2m:co2m@127.0.0.1:5432/co2m"))
    parser.add_argument("--force", action="store_true", help="Replace an existing target schema")
    args = parser.parse_args()

    with psycopg.connect(args.dsn, row_factory=dict_row) as connection:
        exists = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s) AS exists",
            (args.schema,),
        ).fetchone()["exists"]
        if exists and not args.force:
            raise SystemExit(f"Target schema '{args.schema}' already exists. Re-run with --force to replace it.")

        with connection.transaction():
            if exists:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("CREATE TABLE {}.documents (LIKE public.documents INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("CREATE TABLE {}.chunks (LIKE public.chunks INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("ALTER TABLE {}.documents ADD PRIMARY KEY (document_id)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("ALTER TABLE {}.documents ADD UNIQUE (source_path)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("ALTER TABLE {}.chunks ADD PRIMARY KEY (chunk_id)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("ALTER TABLE {}.chunks ADD UNIQUE (document_id, chunk_number)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("ALTER TABLE {}.chunks ADD CONSTRAINT chunks_document_id_fkey FOREIGN KEY (document_id) REFERENCES {}.documents(document_id) ON DELETE CASCADE").format(sql.Identifier(args.schema), sql.Identifier(args.schema)))
            connection.execute(sql.SQL("INSERT INTO {}.documents SELECT * FROM public.documents WHERE is_unique = true").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("INSERT INTO {}.chunks SELECT c.* FROM public.chunks c JOIN {}.documents d ON d.document_id = c.document_id").format(sql.Identifier(args.schema), sql.Identifier(args.schema)))
            connection.execute(sql.SQL("CREATE INDEX chunks_document_idx ON {}.chunks (document_id)").format(sql.Identifier(args.schema)))
            connection.execute(sql.SQL("CREATE INDEX chunks_embedding_idx ON {}.chunks USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL").format(sql.Identifier(args.schema)))

        print({"schema": args.schema, **count_rows(connection, args.schema)})


if __name__ == "__main__":
    main()