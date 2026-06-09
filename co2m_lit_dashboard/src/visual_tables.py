"""
visual_tables.py
----------------
Pre-compute all visualisation-ready tables that the dashboard will consume.

All functions accept DataFrames produced by earlier pipeline stages and
return new DataFrames that are saved as Parquet artifacts.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Concept counts
# ---------------------------------------------------------------------------

def build_concept_counts(concepts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with one row per concept showing how many unique
    chunks and documents were assigned to it.

    Columns: concept_name, chunk_count, doc_count
    """
    if concepts_df.empty:
        return pd.DataFrame(columns=["concept_name", "chunk_count", "doc_count"])

    result = (
        concepts_df.groupby("concept_name")
        .agg(
            chunk_count=("chunk_id", "nunique"),
            doc_count=("doc_id", "nunique"),
        )
        .reset_index()
        .sort_values("chunk_count", ascending=False)
    )
    return result


# ---------------------------------------------------------------------------
# Top documents per concept
# ---------------------------------------------------------------------------

def build_top_documents_by_concept(
    concepts_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the *top_n* documents (by matching chunk count) for every concept.

    Columns: concept_name, doc_id, filename, title, year, chunk_count, rank
    """
    if concepts_df.empty or documents_df.empty:
        return pd.DataFrame(
            columns=[
                "concept_name", "doc_id", "filename",
                "title", "year", "chunk_count", "rank",
            ]
        )

    doc_counts = (
        concepts_df.groupby(["concept_name", "doc_id"])
        .agg(chunk_count=("chunk_id", "nunique"))
        .reset_index()
    )

    doc_counts["rank"] = (
        doc_counts.groupby("concept_name")["chunk_count"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    top = doc_counts[doc_counts["rank"] <= top_n].copy()

    meta_cols = [c for c in ("doc_id", "filename", "title", "year") if c in documents_df.columns]
    top = top.merge(documents_df[meta_cols], on="doc_id", how="left")

    return top.sort_values(["concept_name", "rank"])


# ---------------------------------------------------------------------------
# TF-IDF keywords per concept
# ---------------------------------------------------------------------------

def build_keywords_by_concept(
    concepts_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    taxonomy: Dict[str, List[str]],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract the top *top_n* TF-IDF keywords for the corpus of chunks
    assigned to each concept.

    Columns: concept_name, keyword, weight
    """
    records: list[dict] = []

    for concept_name in taxonomy:
        if concepts_df.empty:
            continue
        chunk_ids = concepts_df.loc[
            concepts_df["concept_name"] == concept_name, "chunk_id"
        ].unique()
        if len(chunk_ids) == 0:
            continue

        texts = chunks_df.loc[
            chunks_df["chunk_id"].isin(chunk_ids), "chunk_text"
        ].tolist()

        if not texts:
            continue

        try:
            vec = TfidfVectorizer(
                max_features=top_n,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            matrix = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()
            mean_scores = np.asarray(matrix.mean(axis=0)).flatten()
            for kw, score in zip(feature_names, mean_scores):
                records.append(
                    {
                        "concept_name": concept_name,
                        "keyword": kw,
                        "weight": round(float(score), 6),
                    }
                )
        except Exception as exc:
            print(f"  [keywords] WARNING – {concept_name}: {exc}")

    if not records:
        return pd.DataFrame(columns=["concept_name", "keyword", "weight"])

    return (
        pd.DataFrame(records)
        .sort_values(["concept_name", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
