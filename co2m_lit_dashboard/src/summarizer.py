"""
summarizer.py
-------------
Purely extractive summarization – no external API or LLM calls.

For each concept, the top-scoring sentences are selected by counting how
many of the concept's keywords appear in that sentence, then normalised by
sentence length (to avoid rewarding very long sentences disproportionately).
"""

import re
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")


def _split_sentences(text: str) -> List[str]:
    """Naive sentence splitter; filters out very short fragments."""
    raw = _SENT_BOUNDARY.split(text)
    return [s.strip() for s in raw if len(s.strip()) > 25]


def _score_sentence(sentence: str, keywords: List[str]) -> float:
    """
    Score = keyword_hit_count / sqrt(word_count).

    Penalising long sentences slightly avoids picking verbose boilerplate.
    """
    norm = sentence.lower()
    hits = sum(1 for kw in keywords if kw in norm)
    if hits == 0:
        return 0.0
    word_count = max(len(sentence.split()), 1)
    return hits / (word_count ** 0.5)


# ---------------------------------------------------------------------------
# Extractive summary builder
# ---------------------------------------------------------------------------

def extractive_summary(
    texts: List[str],
    keywords: List[str],
    n_sentences: int = 5,
) -> str:
    """
    Build an extractive summary from a list of text passages.

    Deduplicates near-identical sentence starts before ranking.
    Falls back to the first few sentences when no keyword matches exist.
    """
    all_sentences: List[str] = []
    for text in texts:
        all_sentences.extend(_split_sentences(text))

    if not all_sentences:
        return "No relevant text found for this concept."

    # Deduplicate by 80-char prefix
    seen: set[str] = set()
    unique: List[str] = []
    for sent in all_sentences:
        key = sent[:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(sent)

    scored = [(s, _score_sentence(s, keywords)) for s in unique]
    scored = [(s, sc) for s, sc in scored if sc > 0]

    if not scored:
        # No keyword hits – just return the first few sentences
        return " ".join(unique[:n_sentences])

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in scored[:n_sentences]]
    return " [...] ".join(top)


# ---------------------------------------------------------------------------
# Concept-level summary table builder
# ---------------------------------------------------------------------------

def build_concept_summaries(
    concepts_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    taxonomy: Dict[str, List[str]],
    n_sentences: int = 5,
) -> List[dict]:
    """
    Return a list of summary dicts (one per concept) ready for conversion
    to a DataFrame / Parquet.
    """
    summaries: List[dict] = []

    for concept_name, keywords in taxonomy.items():
        mask = concepts_df["concept_name"] == concept_name if not concepts_df.empty else pd.Series([], dtype=bool)
        concept_rows = concepts_df[mask] if not concepts_df.empty else pd.DataFrame()

        if concept_rows.empty:
            summaries.append(
                {
                    "concept_name": concept_name,
                    "summary_text": "No documents found for this concept.",
                    "source_doc_count": 0,
                    "source_chunk_count": 0,
                }
            )
            continue

        chunk_ids = concept_rows["chunk_id"].unique()
        texts = (
            chunks_df.loc[chunks_df["chunk_id"].isin(chunk_ids), "chunk_text"]
            .tolist()
        )

        summaries.append(
            {
                "concept_name": concept_name,
                "summary_text": extractive_summary(texts, keywords, n_sentences),
                "source_doc_count": int(concept_rows["doc_id"].nunique()),
                "source_chunk_count": int(len(chunk_ids)),
            }
        )

    return summaries
