"""
storage.py
----------
Save and load pipeline artifacts (Parquet files + manifest JSON).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd


# Canonical artifact names (without extension)
ARTIFACT_NAMES = [
    "documents",
    "pages",
    "chunks",
    "concepts",
    "summaries",
    "concept_counts",
    "top_documents_by_concept",
    "keywords_by_concept",
]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_artifacts(artifacts: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write each DataFrame in *artifacts* as a Parquet file under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in artifacts.items():
        dest = out_dir / f"{name}.parquet"
        df.to_parquet(dest, index=False)
        print(f"  Saved {dest}  ({len(df):,} rows)")


def save_manifest(artifacts: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write a manifest.json summarising the build into *out_dir*."""
    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_files": [f"{k}.parquet" for k in artifacts],
        "row_counts": {k: len(v) for k, v in artifacts.items()},
        "pdf_count": len(artifacts.get("documents", pd.DataFrame())),
        "chunk_count": len(artifacts.get("chunks", pd.DataFrame())),
        "concept_assignment_count": len(artifacts.get("concepts", pd.DataFrame())),
    }
    dest = out_dir / "manifest.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Saved {dest}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_artifacts(artifacts_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all Parquet artifacts from *artifacts_dir*.

    Missing files are silently skipped; the caller receives an empty
    DataFrame for that key.
    """
    result: Dict[str, pd.DataFrame] = {}
    for name in ARTIFACT_NAMES:
        path = artifacts_dir / f"{name}.parquet"
        if path.exists():
            result[name] = pd.read_parquet(path)
        else:
            result[name] = pd.DataFrame()
    return result


def load_manifest(artifacts_dir: Path) -> dict:
    """Load manifest.json; return empty dict if not found."""
    path = artifacts_dir / "manifest.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}
