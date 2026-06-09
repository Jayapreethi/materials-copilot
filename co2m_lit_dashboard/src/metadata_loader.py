"""
metadata_loader.py
------------------
Load external metadata files (CSV, JSON, JSONL) and match records
to PDF filenames so the build pipeline can enrich document-level data.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_metadata_dir(metadata_dir: Path) -> pd.DataFrame:
    """
    Read all CSV / JSON / JSONL files in *metadata_dir* and return a
    single concatenated DataFrame.  Columns are lowercased and stripped.
    """
    frames: list[pd.DataFrame] = []

    for filepath in sorted(metadata_dir.iterdir()):
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(filepath, low_memory=False)
                frames.append(df)
            elif suffix == ".json":
                with open(filepath, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    frames.append(pd.DataFrame(data))
                elif isinstance(data, dict):
                    frames.append(pd.DataFrame([data]))
            elif suffix == ".jsonl":
                frames.append(pd.read_json(filepath, lines=True))
        except Exception as exc:
            print(f"  [metadata] WARNING – could not load {filepath.name}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [str(c).lower().strip() for c in combined.columns]
    return combined


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Return lowercase alphanumeric slug for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def match_metadata_to_doc(
    filename: str,
    metadata_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Try to find a matching row in *metadata_df* for the given PDF *filename*.

    Matching priority:
      1. Exact filename stem match against columns named 'filename', 'file',
         'pdf', or 'name'.
      2. Partial title slug match against a 'title' column.

    Returns a dict with keys: title, authors, year (values may be None).
    """
    empty: Dict[str, Any] = {"title": None, "authors": None, "year": None}

    if metadata_df is None or metadata_df.empty:
        return empty

    stem_slug = _slug(Path(filename).stem)

    # --- Pass 1: filename column match ---
    for col in ("filename", "file", "pdf", "name"):
        if col not in metadata_df.columns:
            continue
        mask = metadata_df[col].astype(str).apply(
            lambda x: _slug(Path(x).stem)
        ) == stem_slug
        if mask.any():
            return _extract_fields(metadata_df.loc[mask].iloc[0])

    # --- Pass 2: title slug containment ---
    if "title" in metadata_df.columns and stem_slug:
        for _, row in metadata_df.iterrows():
            title_slug = _slug(str(row.get("title", "")))
            if title_slug and (
                stem_slug in title_slug or title_slug in stem_slug
            ):
                return _extract_fields(row)

    return empty


def _extract_fields(row: pd.Series) -> Dict[str, Any]:
    """Pull title / authors / year from a metadata row, handling common aliases."""
    result: Dict[str, Any] = {"title": None, "authors": None, "year": None}

    # Title
    for col in ("title", "paper_title", "name", "article_title"):
        val = row.get(col)
        if val is not None and pd.notna(val) and str(val).strip():
            result["title"] = str(val).strip()
            break

    # Authors
    for col in ("authors", "author", "contributors", "creator"):
        val = row.get(col)
        if val is not None and pd.notna(val) and str(val).strip():
            result["authors"] = str(val).strip()
            break

    # Year  – accept integer, float, or string containing a 4-digit year
    for col in ("year", "publication_year", "pub_year", "date", "pub_date", "published"):
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        match = re.search(r"\b(19|20)\d{2}\b", str(val))
        if match:
            result["year"] = int(match.group())
            break

    return result
