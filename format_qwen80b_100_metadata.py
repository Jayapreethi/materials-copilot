"""Convert the 100-record Qwen JSONL batch to an indented Sloan schema export."""

import hashlib
import json
import re
from pathlib import Path

from export_qwen80b_metadata import convert_record


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "co2m" / "metadata" / "qwen3_next_80b_100_metadata.jsonl"
OUTPUT_PATH = ROOT / "co2m" / "metadata" / "qwen80b_100_pdfs_metadata.json"
LOCAL_ID = re.compile(r"^(TECH|SCALE|EVID|FACT|TRANS|CHAL|OUT)_\d{3}$")


def replace_ids(value, replacements):
    if isinstance(value, dict):
        return {key: replace_ids(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_ids(item, replacements) for item in value]
    return replacements.get(value, value)


def source_name(filename):
    return filename.partition("_")[0].upper() or "UNKNOWN"


def normalize_ids(record):
    document = record["document"]
    source = source_name(document["filename"])
    digest = document["content_sha256"][:12].upper()
    filename_digest = hashlib.sha256(document["filename"].encode("utf-8")).hexdigest()[:8].upper()
    identity = f"{digest}_{filename_digest}"
    document["document_id"] = f"DOC_{source}_{identity}"
    document["discovery_source"] = source

    title = (document.get("title") or "").lower()
    topic = record["taxonomy"]["primary_topic"]
    if "calcium looping" in title:
        topic["core_id"] = "CCUS_CAPTURE"
    elif "electroreduction" in title or "electrochemical co2" in title:
        topic["core_id"] = "CO2_Conversion"

    replacements = {}
    counts = {}
    for identifier in LOCAL_ID.findall(json.dumps(record)):
        prefix = identifier
        counts[prefix] = counts.get(prefix, 0) + 1
        old = f"{prefix}_{counts[prefix]:03d}"
        replacements[old] = f"{prefix}_{source}_{identity}_{counts[prefix]:03d}"
    return replace_ids(record, replacements)


def main():
    source_records = [
        json.loads(line)
        for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unique_records = {}
    for item in source_records:
        unique_records.setdefault(item["filename"].lower(), item)
    records = [
        normalize_ids(convert_record(item, index))
        for index, item in enumerate(unique_records.values(), start=1)
    ]
    if len(records) != 100:
        raise ValueError(f"Expected 100 unique records; found {len(records)}")
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT_PATH} with {len(records)} formatted records.")


if __name__ == "__main__":
    main()