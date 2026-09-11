#!/usr/bin/env python
"""
Export Postgres chunks (+ joined document fields) to Parquet files.
Read-only: does not modify anything in Postgres.

Produces:
    co2m/artifacts/chunks_full_backup.parquet  - all chunks (current full corpus)
    co2m/artifacts/chunks_unique.parquet       - only chunks from is_unique=true documents

Usage:
    python export_chunks.py
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row

DSN = os.getenv("CO2M_POSTGRES_DSN", "postgresql://co2m:co2m@127.0.0.1:5432/co2m")
BATCH_SIZE = 5000
OUTPUT_DIR = Path("co2m/artifacts")

QUERY_TEMPLATE = """
    SELECT c.chunk_id, c.document_id, d.filename, d.title, d.publication_year,
           d.is_unique, c.chunk_number, c.page_start, c.page_end, c.content,
           c.embedding::text AS embedding_text
    FROM chunks c
    JOIN documents d ON d.document_id = c.document_id
    {where}
    ORDER BY c.chunk_id
"""


def parse_embedding(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(v) for v in text.strip("[]").split(",")]


SCHEMA = pa.schema([
    ("chunk_id", pa.int64()),
    ("document_id", pa.int64()),
    ("filename", pa.string()),
    ("title", pa.string()),
    ("publication_year", pa.int64()),
    ("is_unique", pa.bool_()),
    ("chunk_number", pa.int64()),
    ("page_start", pa.int64()),
    ("page_end", pa.int64()),
    ("content", pa.string()),
    ("embedding", pa.list_(pa.float32())),
])


def export(output_path: Path, unique_only: bool) -> int:
    where = "WHERE d.is_unique = true" if unique_only else ""
    query = QUERY_TEMPLATE.format(where=where)
    total = 0
    writer = None
    with psycopg.connect(DSN, row_factory=dict_row) as connection:
        with connection.cursor(name="export_chunks_cursor") as cursor:
            cursor.itersize = BATCH_SIZE
            cursor.execute(query)
            while True:
                rows = cursor.fetchmany(BATCH_SIZE)
                if not rows:
                    break
                for row in rows:
                    row["embedding"] = parse_embedding(row.pop("embedding_text"))
                table = pa.Table.from_pylist(rows, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_path), SCHEMA)
                writer.write_table(table)
                total += len(rows)
                print(f"  {output_path.name}: wrote {total} rows", flush=True)
    if writer is not None:
        writer.close()
    return total


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_path = OUTPUT_DIR / "chunks_full_backup.parquet"
    print(f"Exporting full chunk backup -> {full_path}")
    full_count = export(full_path, unique_only=False)

    unique_path = OUTPUT_DIR / "chunks_unique.parquet"
    print(f"Exporting unique-only chunks -> {unique_path}")
    unique_count = export(unique_path, unique_only=True)

    print(f"Done. full={full_count} unique={unique_count}")


if __name__ == "__main__":
    main()
