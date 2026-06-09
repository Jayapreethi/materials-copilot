"""
pdf_extractor.py
----------------
Extract text and metadata from PDF files using PyMuPDF.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def compute_file_hash(filepath: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# PDF metadata extraction
# ---------------------------------------------------------------------------

def _parse_year_from_pdf_date(date_str: str) -> int | None:
    """Try to parse a 4-digit year from a PDF date string (e.g. 'D:20210315...')."""
    if not date_str:
        return None
    # PDF dates are typically: D:YYYYMMDDHHmmSSOHH'mm'
    match = re.search(r"(19|20)\d{2}", date_str)
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass
    return None


def _extract_pdf_meta(doc: fitz.Document) -> Dict[str, Any]:
    """Pull title, authors, and year from a PDF's internal metadata."""
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip() or None
    authors = (meta.get("author") or "").strip() or None
    year = _parse_year_from_pdf_date(meta.get("creationDate", ""))
    return {"title": title, "authors": authors, "year": year}


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_pages(filepath: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Open a PDF and return (doc_info, pages).

    doc_info keys: filename, title, authors, year, file_hash, page_count
    pages list items: {page_number, text}
    """
    doc = fitz.open(str(filepath))
    meta = _extract_pdf_meta(doc)
    file_hash = compute_file_hash(filepath)

    pages: List[Dict[str, Any]] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text") or ""
        # Collapse excessive whitespace while preserving sentence boundaries
        cleaned = re.sub(r"[ \t]{2,}", " ", raw_text).strip()
        pages.append({"page_number": page_num + 1, "text": cleaned})

    doc_info: Dict[str, Any] = {
        "filename": filepath.name,
        "title": meta["title"],
        "authors": meta["authors"],
        "year": meta["year"],
        "file_hash": file_hash,
        "page_count": len(doc),
    }
    doc.close()
    return doc_info, pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into word-count-bounded chunks with a sliding overlap.

    Parameters
    ----------
    text       : Source text string.
    chunk_size : Target number of words per chunk.
    overlap    : Number of words to repeat at the start of the next chunk.

    Returns
    -------
    List of non-empty chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks
