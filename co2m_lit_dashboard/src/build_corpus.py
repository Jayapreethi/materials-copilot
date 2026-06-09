#!/usr/bin/env python3
"""
build_corpus.py
---------------
Entry point for the CO2M Literature Intelligence build pipeline.

Usage (run from workspace root):
    python co2m_lit_dashboard/src/build_corpus.py \
        --pdf-dir     co2m/pdfs \
        --metadata-dir co2m/metadata \
        --out          co2m/artifacts

All output artifacts are written to --out as Parquet files plus a
manifest.json.  The dashboard reads only from that directory and does not
need the original PDFs.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow imports from this directory regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from concept_classifier import classify_all_chunks, load_taxonomy
from metadata_loader import load_metadata_dir, match_metadata_to_doc
from pdf_extractor import extract_pages, split_into_chunks
from storage import save_artifacts, save_manifest
from summarizer import build_concept_summaries
from visual_tables import (
    build_concept_counts,
    build_keywords_by_concept,
    build_top_documents_by_concept,
)


# ---------------------------------------------------------------------------
# Default config path (relative to this script)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "concepts.yaml"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_corpus(
    pdf_dir: Path,
    metadata_dir: Path,
    out_dir: Path,
    config_path: Path,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("CO2M Literature Intelligence Pipeline  –  Build Phase")
    print(sep)
    print(f"  PDF directory      : {pdf_dir.resolve()}")
    print(f"  Metadata directory : {metadata_dir.resolve()}")
    print(f"  Output directory   : {out_dir.resolve()}")
    print(f"  Config             : {config_path.resolve()}")

    # ------------------------------------------------------------------
    # 1. Taxonomy
    # ------------------------------------------------------------------
    print("\n[1/7] Loading concept taxonomy …")
    taxonomy = load_taxonomy(config_path)
    print(f"  {len(taxonomy)} concepts: {list(taxonomy.keys())}")

    # ------------------------------------------------------------------
    # 2. External metadata
    # ------------------------------------------------------------------
    print("\n[2/7] Loading external metadata …")
    metadata_df: pd.DataFrame = pd.DataFrame()
    if metadata_dir.exists():
        metadata_df = load_metadata_dir(metadata_dir)
        print(f"  {len(metadata_df):,} metadata records loaded")
    else:
        print(f"  Directory not found – skipping ({metadata_dir})")

    # ------------------------------------------------------------------
    # 3. PDF discovery
    # ------------------------------------------------------------------
    print("\n[3/7] Discovering PDFs …")
    if not pdf_dir.exists():
        print(f"  WARNING: PDF directory does not exist: {pdf_dir}")
        pdf_files: list[Path] = []
    else:
        pdf_files = sorted(
            p for p in pdf_dir.iterdir()
            if p.suffix.lower() == ".pdf"
        )
        print(f"  Found {len(pdf_files)} PDF file(s)")

    # ------------------------------------------------------------------
    # 4. Extract text → documents / pages / chunks
    # ------------------------------------------------------------------
    documents: list[dict] = []
    pages: list[dict] = []
    chunks: list[dict] = []

    doc_id = page_id = chunk_id = 0

    for pdf_path in pdf_files:
        print(f"  → {pdf_path.name}")
        try:
            doc_info, doc_pages = extract_pages(pdf_path)
        except Exception as exc:
            print(f"    ERROR extracting {pdf_path.name}: {exc}")
            continue

        # Enrich with external metadata (non-destructive – only fills blanks)
        ext = match_metadata_to_doc(pdf_path.name, metadata_df)
        doc_info["title"] = doc_info["title"] or ext["title"] or pdf_path.stem
        doc_info["authors"] = doc_info["authors"] or ext["authors"]
        doc_info["year"] = doc_info["year"] or ext["year"]
        doc_info["doc_id"] = doc_id
        documents.append(doc_info)

        for page in doc_pages:
            pages.append(
                {
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "filename": pdf_path.name,
                    "page_number": page["page_number"],
                    "text": page["text"],
                }
            )
            page_id += 1

            for chunk_text in split_into_chunks(page["text"]):
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "filename": pdf_path.name,
                        "page_number": page["page_number"],
                        "chunk_text": chunk_text,
                    }
                )
                chunk_id += 1

        doc_id += 1

    documents_df = pd.DataFrame(documents)
    pages_df = pd.DataFrame(pages)
    chunks_df = pd.DataFrame(chunks)

    print(
        f"\n  Totals → {len(documents_df)} doc(s) | "
        f"{len(pages_df):,} page(s) | {len(chunks_df):,} chunk(s)"
    )

    # ------------------------------------------------------------------
    # 5. Concept classification
    # ------------------------------------------------------------------
    print("\n[4/7] Classifying chunks by concept …")
    _empty_concepts_cols = [
        "concept_id", "concept_name", "doc_id", "chunk_id",
        "filename", "page_number", "confidence_score",
    ]
    if not chunks_df.empty:
        concept_records = classify_all_chunks(chunks_df, taxonomy, min_confidence=0.05)
        concepts_df = pd.DataFrame(concept_records) if concept_records else pd.DataFrame(columns=_empty_concepts_cols)
    else:
        concepts_df = pd.DataFrame(columns=_empty_concepts_cols)
    print(f"  {len(concepts_df):,} concept assignment(s)")

    # ------------------------------------------------------------------
    # 6. Extractive summaries
    # ------------------------------------------------------------------
    print("\n[5/7] Building extractive concept summaries …")
    summary_records = build_concept_summaries(concepts_df, chunks_df, taxonomy)
    summaries_df = pd.DataFrame(summary_records)
    print(f"  {len(summaries_df)} concept summaries built")

    # ------------------------------------------------------------------
    # 7. Visualisation tables
    # ------------------------------------------------------------------
    print("\n[6/7] Pre-computing visualisation tables …")
    concept_counts_df = build_concept_counts(concepts_df)
    top_docs_df = build_top_documents_by_concept(concepts_df, documents_df)
    keywords_df = build_keywords_by_concept(concepts_df, chunks_df, taxonomy)
    print(f"  concept_counts              : {len(concept_counts_df):>6,} rows")
    print(f"  top_documents_by_concept    : {len(top_docs_df):>6,} rows")
    print(f"  keywords_by_concept         : {len(keywords_df):>6,} rows")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    print(f"\n[7/7] Saving artifacts → {out_dir} …")
    artifacts = {
        "documents": documents_df,
        "pages": pages_df,
        "chunks": chunks_df,
        "concepts": concepts_df,
        "summaries": summaries_df,
        "concept_counts": concept_counts_df,
        "top_documents_by_concept": top_docs_df,
        "keywords_by_concept": keywords_df,
    }
    save_artifacts(artifacts, out_dir)
    save_manifest(artifacts, out_dir)

    print(f"\n{sep}")
    print("Build complete!")
    print(f"Artifacts written to: {out_dir.resolve()}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CO2M literature intelligence artifacts from PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("co2m/pdfs"),
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("co2m/metadata"),
        help="Directory containing metadata CSV/JSON files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("co2m/artifacts"),
        help="Output directory for Parquet artifacts",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to concepts.yaml taxonomy config",
    )
    args = parser.parse_args()
    build_corpus(args.pdf_dir, args.metadata_dir, args.out, args.config)


if __name__ == "__main__":
    main()
