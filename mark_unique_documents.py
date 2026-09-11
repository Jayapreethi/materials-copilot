#!/usr/bin/env python
"""
Non-destructively flag documents.is_unique = true for filenames present in
a metadata-derived unique-file list (e.g. co2m/metadata/qwen80b_10903_unique_pdf_filenames.txt).
Nothing is deleted; documents/chunks not in the list simply stay is_unique = false.

Usage:
    python mark_unique_documents.py [--list co2m/metadata/qwen80b_10903_unique_pdf_filenames.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vector_db.postgres_corpus_store import PostgresCorpusStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        type=Path,
        default=Path("co2m/metadata/qwen80b_10903_unique_pdf_filenames.txt"),
    )
    parser.add_argument("--dsn", default=None)
    args = parser.parse_args()

    filenames = {line.strip() for line in args.list.read_text(encoding="utf-8").splitlines() if line.strip()}
    store = PostgresCorpusStore(dsn=args.dsn)
    result = store.mark_unique_documents(filenames)
    print(result)
    print(store.stats())


if __name__ == "__main__":
    main()
