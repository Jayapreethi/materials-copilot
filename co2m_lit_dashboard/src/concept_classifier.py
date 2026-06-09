"""
concept_classifier.py
---------------------
Keyword-based concept classification for text chunks.

Each chunk is tested against every concept in the taxonomy; a confidence
score is computed as (unique keyword hits) / (total keywords for concept).
Only assignments above *min_confidence* are kept.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def load_taxonomy(config_path: Path) -> Dict[str, List[str]]:
    """Load concept → keyword-list mapping from a YAML file."""
    with open(config_path, encoding="utf-8") as fh:
        taxonomy = yaml.safe_load(fh)
    # Ensure all keyword lists are lowercased strings
    return {
        concept: [str(kw).lower().strip() for kw in keywords]
        for concept, keywords in taxonomy.items()
        if keywords
    }


# ---------------------------------------------------------------------------
# Single-chunk classification
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def classify_chunk(
    chunk_text: str,
    taxonomy: Dict[str, List[str]],
    min_confidence: float = 0.05,
) -> List[Tuple[str, float]]:
    """
    Return a sorted list of (concept_name, confidence_score) for *chunk_text*.

    confidence_score = unique_keyword_hits / total_keywords_in_concept
    """
    norm = _normalize(chunk_text)
    results: List[Tuple[str, float]] = []

    for concept_name, keywords in taxonomy.items():
        if not keywords:
            continue
        hits = sum(
            1
            for kw in keywords
            if re.search(r"\b" + re.escape(kw) + r"\b", norm)
        )
        if hits == 0:
            continue
        confidence = round(hits / len(keywords), 4)
        if confidence >= min_confidence:
            results.append((concept_name, confidence))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def classify_all_chunks(
    chunks_df: pd.DataFrame,
    taxonomy: Dict[str, List[str]],
    min_confidence: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Classify every row in *chunks_df* and return a list of concept-assignment
    dicts matching the ``concepts`` schema.

    Expected columns in *chunks_df*: chunk_id, doc_id, filename, page_number,
    chunk_text.
    """
    records: List[Dict[str, Any]] = []
    concept_id = 0

    for _, row in chunks_df.iterrows():
        assignments = classify_chunk(row["chunk_text"], taxonomy, min_confidence)
        for concept_name, score in assignments:
            records.append(
                {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "doc_id": row["doc_id"],
                    "chunk_id": row["chunk_id"],
                    "filename": row["filename"],
                    "page_number": row["page_number"],
                    "confidence_score": score,
                }
            )
            concept_id += 1

    return records
