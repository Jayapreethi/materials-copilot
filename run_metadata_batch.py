"""Extract Qwen metadata for any PDF count and write a formatted Sloan JSON export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from export_qwen80b_metadata import convert_record
from format_qwen80b_100_metadata import normalize_ids
from generate_qwen_metadata import ollama_extract, parse_pdf, representative_text


ROOT = Path(__file__).resolve().parent

# Only these indicate the run itself is broken (Ollama down/unreachable). Everything
# else is a per-PDF failure and must not trip the consecutive-error kill switch.
SYSTEMIC_ERRORS = (ConnectionError, urllib.error.URLError, socket.timeout, TimeoutError)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error.msg}") from error
    return records


def get_unique_output_path(path: Path) -> Path:
    """Return path if it doesn't exist; otherwise return a new path with _runN appended before suffix."""
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    run_index = 2
    while True:
        new_path = parent / f"{stem}_run{run_index}{suffix}"
        if not new_path.exists():
            return new_path
        run_index += 1


def write_issue(issue_path: Path, pdf_path: Path, error: Exception) -> dict:
    issue = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": pdf_path.name,
        "source_path": pdf_path.as_posix(),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    with issue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(issue, ensure_ascii=False) + "\n")
    return issue


def write_issue_summary(path: Path, issues: list[dict], cause_of_ending: str = "completed") -> None:
    columns = ["timestamp", "filename", "source_path", "error_type", "error_message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(issues, key=lambda issue: issue["filename"].lower()))


def format_export(raw_path: Path, formatted_path: Path, expected_count: int, allow_overwrite: bool = False) -> tuple[Path, int]:
    actual_path = formatted_path if allow_overwrite else get_unique_output_path(formatted_path)
    unique_records = {}
    for record in load_jsonl(raw_path):
        unique_records.setdefault(record["filename"].lower(), record)
    formatted_records = [
        normalize_ids(convert_record(record, index))
        for index, record in enumerate(unique_records.values(), start=1)
    ]
    actual_path.write_text(
        json.dumps(formatted_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if len(formatted_records) > expected_count:
        raise ValueError("Formatted export contains more records than the selected PDF count")
    return actual_path, len(formatted_records)


def check_ollama_server(url: str, model: str | None = None) -> None:
    endpoint = url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(endpoint)
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.load(response)
            if model and "models" in payload:
                available = [m.get("name", "") for m in payload.get("models", [])]
                if model not in available and f"{model}:latest" not in available and not any(m.startswith(model) for m in available):
                    print(f"Warning: Model '{model}' not explicitly listed in Ollama tags. Found: {available}", flush=True)
    except Exception as error:
        raise ConnectionError(
            f"Cannot connect to Ollama server at '{url}'.\n"
            f"Error details: {error}\n"
            f"Fix: Make sure Ollama service is running (e.g. run 'ollama serve' or start the Ollama desktop app)."
        ) from error


def extract_pdf(pdf_path: Path, args: argparse.Namespace) -> dict:
    pages = parse_pdf(pdf_path)
    text, source_pages = representative_text(pages, args.max_chars)
    if not text:
        raise ValueError("PDF contains no extractable text")

    for attempt in range(args.retries + 1):
        try:
            metadata = ollama_extract(
                args.ollama_url,
                args.model,
                text,
                args.timeout,
                args.context_length,
                args.num_predict,
            )
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            if attempt == args.retries:
                raise
            time.sleep(2**attempt)

    return {
        "filename": pdf_path.name,
        "source_path": pdf_path.as_posix(),
        "content_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pages": source_pages,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Qwen metadata for a selected number of PDFs and create formatted JSON."
    )
    parser.add_argument("--count", type=int, required=True, help="Number of sorted PDFs to process, e.g. 5, 10, 100, or 10000.")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "all_pdfs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "co2m" / "metadata")
    parser.add_argument("--name", help="Artifact name stem. Defaults to qwen80b_<count>_pdfs_metadata.")
    parser.add_argument("--raw-input", type=Path, help="Existing JSONL metadata to format instead of the default raw-output path.")
    parser.add_argument("--format-only", action="store_true", help="Skip extraction and only create the formatted JSON and issue reports.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing formatted JSON file instead of creating a new file.")
    parser.add_argument("--max-consecutive-errors", type=int, default=10, help="Stop run after this many consecutive connection/timeout errors (0 for unlimited). Per-PDF failures are skipped and logged instead.")
    parser.add_argument("--model", default="qwen3-next:80b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-chars", type=int, default=60000)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--num-predict", type=int, default=8192, help="Max output tokens for the model response; increase if JSON responses are truncated.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    selected_pdfs = sorted(args.pdf_dir.rglob("*.pdf"))[:args.count]
    if len(selected_pdfs) < args.count:
        parser.error(f"Only {len(selected_pdfs)} PDFs found in {args.pdf_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"qwen80b_{args.count}_pdfs_metadata"
    raw_path = args.raw_input or args.output_dir / f"{name}.jsonl"
    formatted_path = args.output_dir / f"{name}.json"
    issues_jsonl_path = args.output_dir / f"{name}_issues.jsonl"
    issues_csv_path = args.output_dir / f"{name}_issues.csv"
    issues_jsonl_path.touch(exist_ok=True)

    completed = {record["filename"].lower() for record in load_jsonl(raw_path)}
    pending = [] if args.format_only else [path for path in selected_pdfs if path.name.lower() not in completed]
    run_issues = []
    print(f"selected={len(selected_pdfs)} completed={len(completed)} pending={len(pending)}")

    cause_of_ending = "completed"
    consecutive_errors = 0

    if pending:
        check_ollama_server(args.ollama_url, args.model)
        with raw_path.open("a", encoding="utf-8") as output:
            for index, pdf_path in enumerate(pending, start=1):
                try:
                    record = extract_pdf(pdf_path, args)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    consecutive_errors = 0
                    print(f"[{index}/{len(pending)}] {pdf_path.name}: ok", flush=True)
                except KeyboardInterrupt:
                    cause_of_ending = "interrupted_by_user"
                    print(f"\n[{index}/{len(pending)}] Run interrupted by user (KeyboardInterrupt).", flush=True)
                    break
                except Exception as error:
                    if isinstance(error, SYSTEMIC_ERRORS):
                        consecutive_errors += 1
                    run_issues.append(write_issue(issues_jsonl_path, pdf_path, error))
                    print(f"[{index}/{len(pending)}] {pdf_path.name}: {type(error).__name__}: {error}", flush=True)
                    if args.max_consecutive_errors > 0 and consecutive_errors >= args.max_consecutive_errors:
                        cause_of_ending = f"stopped_after_{consecutive_errors}_consecutive_connection_errors"
                        print(f"\nStopping run due to {consecutive_errors} consecutive connection errors.", flush=True)
                        break

    completed = {record["filename"].lower() for record in load_jsonl(raw_path)}
    missing = [path for path in selected_pdfs if path.name.lower() not in completed]
    for pdf_path in missing:
        if not any(issue["filename"].lower() == pdf_path.name.lower() for issue in run_issues):
            run_issues.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filename": pdf_path.name,
                "source_path": pdf_path.as_posix(),
                "error_type": "IncompleteRun",
                "error_message": f"No completed metadata record is present for this selected PDF. Cause of ending: {cause_of_ending}",
            })
    write_issue_summary(issues_csv_path, run_issues, cause_of_ending=cause_of_ending)

    actual_formatted_path, formatted_count = format_export(
        raw_path, formatted_path, args.count, allow_overwrite=args.overwrite
    )
    print(f"cause_of_ending={cause_of_ending}")
    print(f"formatted_records={formatted_count} missing={len(missing)}")
    print(f"raw_jsonl={raw_path}")
    print(f"formatted_json={actual_formatted_path}")
    print(f"issues_csv={issues_csv_path}")
    print(f"issues_jsonl={issues_jsonl_path}")


if __name__ == "__main__":
    main()