#!/usr/bin/env python
"""Merge the Qwen metadata runs for the 10,903-PDF corpus."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_RUNS = [
    "qwen80b_10903_pdfs_metadata.json",
    "qwen80b_10903_pdfs_metadata_run2.json",
    "qwen80b_10903_pdfs_metadata_run3.json",
    "qwen80b_10903_pdfs_metadata_run4.json",
    "qwen80b_10903_pdfs_metadata_run5.json",
    "qwen80b_10903_pdfs_metadata_run6.json",
    "qwen80b_10903_pdfs_metadata_run7.json",
]


def load_run(path: Path) -> tuple[list[dict], str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [], f"{type(error).__name__}: {error}"
    if not isinstance(data, list):
        return [], f"Expected a JSON list, got {type(data).__name__}"
    return [record for record in data if isinstance(record, dict)], None


def record_key(record: dict) -> str | None:
    document = record.get("document")
    if not isinstance(document, dict):
        return None
    filename = document.get("filename")
    return filename.strip() if isinstance(filename, str) and filename.strip() else None


def merge_runs(input_dir: Path, output_path: Path, report_path: Path, target_count: int) -> dict:
    merged: dict[str, dict] = {}
    duplicate_counts: Counter[str] = Counter()
    run_reports: list[dict] = []
    malformed_records = 0

    for run_name in DEFAULT_RUNS:
        path = input_dir / run_name
        if not path.exists():
            run_reports.append({"file": run_name, "status": "missing", "records": 0})
            continue

        records, error = load_run(path)
        run_reports.append({
            "file": run_name,
            "status": "error" if error else "ok",
            "records": len(records),
            **({"error": error} if error else {}),
        })
        for record in records:
            key = record_key(record)
            if key is None:
                malformed_records += 1
                continue
            if key in merged:
                duplicate_counts[key] += 1
                continue
            merged[key] = record

    output_records = [merged[key] for key in sorted(merged)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    unique_metadata_path = output_path.with_name("qwen80b_10903_unique_pdfs_metadata.json")
    unique_metadata_path.write_text(json.dumps(output_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    unique_list_path = output_path.with_name("qwen80b_10903_unique_pdf_filenames.txt")
    unique_list_path.write_text("\n".join(sorted(merged)) + "\n", encoding="utf-8")

    total_input_records = sum(run["records"] for run in run_reports)
    report = {
        "target_pdf_count": target_count,
        "input_runs": run_reports,
        "total_input_records": total_input_records,
        "unique_pdf_count": len(output_records),
        "duplicate_records_removed": sum(duplicate_counts.values()),
        "duplicate_pdf_names": len(duplicate_counts),
        "malformed_records_skipped": malformed_records,
        "target_difference": len(output_records) - target_count,
        "output_file": str(output_path),
        "unique_metadata_file": str(unique_metadata_path),
        "unique_pdf_list": str(unique_list_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("co2m/metadata"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("co2m/metadata/qwen80b_10903_pdfs_metadata_merged.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("co2m/metadata/qwen80b_10903_pdfs_metadata_merged_report.json"),
    )
    parser.add_argument("--target-count", type=int, default=10_903)
    args = parser.parse_args()
    report = merge_runs(args.metadata_dir, args.output, args.report, args.target_count)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()