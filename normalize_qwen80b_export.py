"""Normalize source labels, canonical taxonomy IDs, and record IDs in the export."""

import json
import re
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parent / "co2m" / "metadata" / "qwen80b_10_pdfs_metadata.json"
ID_PATTERN = re.compile(r"^(TECH|SCALE|EVID|FACT|TRANS|CHAL|OUT)_\d{3}$")


def source_from_filename(filename):
    return filename.split("_", 1)[0].upper() if "_" in filename else "UNKNOWN"


def replace_ids(value, replacements):
    if isinstance(value, dict):
        return {key: replace_ids(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_ids(item, replacements) for item in value]
    return replacements.get(value, value)


def normalize_record(record):
    document = record["document"]
    source = source_from_filename(document["filename"])
    document_hash = document["content_sha256"][:12].upper()
    document["document_id"] = f"DOC_{source}_{document_hash}"
    document["discovery_source"] = source

    taxonomy = record["taxonomy"]["primary_topic"]
    title = document["title"].lower()
    if "calcium looping" in title:
        taxonomy["core_id"] = "CCUS_CAPTURE"
    elif "electroreduction" in title:
        taxonomy["core_id"] = "CO2_Conversion"

    replacements = {}
    counters = {}
    for match in ID_PATTERN.finditer(json.dumps(record)):
        old_id = match.group(0)
        prefix = match.group(1)
        if old_id not in replacements:
            counters[prefix] = counters.get(prefix, 0) + 1
            replacements[old_id] = f"{prefix}_{source}_{document_hash}_{counters[prefix]:03d}"
    return replace_ids(record, replacements)


def main():
    records = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    normalized = [normalize_record(record) for record in records]
    OUTPUT_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Normalized {len(normalized)} records in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()