"""Run the two-pass scale-evidence and scale-derived TRL proxy pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = Path(__file__).resolve().parent
DEFAULT_TARGETS = METADATA_DIR / "qwen80b_10903_unique_pdfs_metadata.json"
DEFAULT_OUTPUT = METADATA_DIR / "scale_trl_two_pass_3899.jsonl"
sys.path.insert(0, str(ROOT))

from generate_qwen_metadata import parse_json_response
from ingest_postgres import parse_pdf

SCALE_PAGE_TERMS = re.compile(
    r"\b(?:scale[- ]?up|scale[- ]?down|laboratory|lab[- ]scale|bench[- ]scale|"
    r"pilot|demonstrat|slipstream|field test|industrial|commercial|prototype|"
    r"reactor|vessel|autoclave|throughput|feed rate|flow rate|capacity|"
    r"continuous operation|operating duration|kg\s*/\s*(?:h|d)|ton\s*/\s*day|"
    r"technology readiness|\btrl\b|batch size|electrode area|membrane area)\b",
    re.IGNORECASE,
)

EXTRACTION_SYSTEM_PROMPT = """You are a scientific information-extraction model supporting a curated process-engineering scale-up knowledge base.

Extract only factual evidence about the physical scale, equipment scale, processing capacity, and experimental maturity actually performed, constructed, tested, or operated in THIS DOCUMENT. Do not judge research quality and do not infer a TRL.

Exclude cited prior studies, background literature, future/proposed work, unrelated supplier specifications, and analytical characterization/sample-preparation quantities. Distinguish primary process quantities from analytical measurements. Never invent quantities or estimate unspecified amounts. Identify the largest primary process experiment. If the document is a review, perspective, commentary, literature survey, or modeling-only and has no original experimental process-scale work, identify that accurately and set exclude_from_scale_classification accordingly. Extract an explicit TRL only when the document states one for the investigated technology, verbatim.

Return only a JSON object matching the supplied schema. Use empty arrays, empty strings, or null when facts are absent. Set evidence_sufficient_for_scale_classification false when the supplied document text does not permit a scale estimate."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "exclude_from_scale_classification": {"type": "boolean"},
        "explicit_trl": {
            "type": "object",
            "properties": {
                "reported": {"type": "boolean"},
                "trl_value": {"type": ["string", "number", "null"]},
                "evidence": {"type": ["string", "null"]},
            },
            "required": ["reported", "trl_value", "evidence"],
            "additionalProperties": False,
        },
        "primary_process": {"type": "string"},
        "quantitative_scale_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": "string"},
                    "context": {"type": "string"},
                    "is_primary_process_scale": {"type": "boolean"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["parameter", "value", "unit", "context", "is_primary_process_scale", "evidence_quote"],
                "additionalProperties": False,
            },
        },
        "equipment_scale_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "equipment": {"type": "string"},
                    "size_or_capacity": {"type": "string"},
                    "context": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["equipment", "size_or_capacity", "context", "evidence_quote"],
                "additionalProperties": False,
            },
        },
        "operating_scale_evidence": {
            "type": "object",
            "properties": {
                "batch_or_continuous": {"type": "string"},
                "throughput": {"type": "string"},
                "operating_duration": {"type": "string"},
                "power_or_capacity": {"type": "string"},
                "integration_description": {"type": "string"},
                "operating_environment": {"type": "string"},
            },
            "required": ["batch_or_continuous", "throughput", "operating_duration", "power_or_capacity", "integration_description", "operating_environment"],
            "additionalProperties": False,
        },
        "scale_terms_found": {"type": "array", "items": {"type": "string"}},
        "largest_demonstrated_scale": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "evidence_quote": {"type": "string"},
            },
            "required": ["description", "evidence_quote"],
            "additionalProperties": False,
        },
        "evidence_sufficient_for_scale_classification": {"type": "boolean"},
        "extraction_notes": {"type": "string"},
    },
    "required": [
        "document_type", "exclude_from_scale_classification", "explicit_trl", "primary_process",
        "quantitative_scale_evidence", "equipment_scale_evidence", "operating_scale_evidence",
        "scale_terms_found", "largest_demonstrated_scale", "evidence_sufficient_for_scale_classification",
        "extraction_notes",
    ],
    "additionalProperties": False,
}

CLASSIFIER_SYSTEM_PROMPT = """You are classifying scientific documents according to the physical and engineering scale demonstrated by the work.

This classification is a SCALE-DERIVED TRL PROXY, not an official Technology Readiness Level. Use only the supplied extracted evidence. Do not introduce outside knowledge. Do not treat a system mentioned in prior literature or proposed future work as operated in this document. Explicit pilot, demonstration, slipstream, field, or industrial terminology is strong evidence only when the extracted record indicates the system was actually operated or tested.

Choose exactly one classification:
- TRL_2_3_PROXY: laboratory concept/proof-of-concept; mg to about 100 g materials, mL to about 1 L process volumes, small laboratory vessels/cells, coupon-scale or small batch experiments.
- TRL_4_5_PROXY: integrated laboratory/bench/engineering validation; about 0.1-10 kg material, 1-100 L process vessels, bench continuous systems, meaningful throughput, integrated units/skids, or sustained engineering operation.
- TRL_6_7_PROXY: pilot/demonstration-relevant operation; tens of kg or more, about 100 L or larger systems, substantial throughput, pilot plant, industrial slipstream, field/site operation, or sustained integrated operation outside simple laboratory conditions.
- INSUFFICIENT_EVIDENCE: not enough extracted evidence to estimate physical or engineering scale.
- REVIEW_EXCLUDED: review/perspective/commentary/meta-analysis or another document without original experimental process-scale work.

These are guidance bands, not exact numeric cutoffs. Prefer insufficient evidence over an unsupported inference. Return only the JSON object matching the supplied schema. Cite concise verbatim evidence quotes from the extracted record in supporting_evidence. Keep explicit TRL separate; do not use it to select the proxy band."""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": ["TRL_2_3_PROXY", "TRL_4_5_PROXY", "TRL_6_7_PROXY", "INSUFFICIENT_EVIDENCE", "REVIEW_EXCLUDED"],
        },
        "classification_rationale": {"type": "string"},
        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["classification", "classification_rationale", "supporting_evidence"],
    "additionalProperties": False,
}


def scale_representative_text(pages: list[tuple[int, str]], max_chars: int) -> tuple[str, list[int]]:
    selected = [(number, text) for number, text in pages if number <= 3 or SCALE_PAGE_TERMS.search(text)]
    output: list[str] = []
    source_pages: list[int] = []
    used = 0
    for page_number, page_text in selected:
        section = f"\n--- PAGE {page_number} ---\n{page_text}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        output.append(section[:remaining])
        source_pages.append(page_number)
        used += min(len(section), remaining)
    return "".join(output), source_pages


def call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    timeout: int,
    context_length: int,
    num_predict: int,
) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "stream": False,
        "think": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": context_length, "num_predict": num_predict},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    parsed = parse_json_response(payload["message"]["content"])
    if not isinstance(parsed, dict):
        raise ValueError("Ollama response was not a JSON object")
    return parsed


def check_ollama(base_url: str, model: str) -> None:
    with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=10) as response:
        payload = json.load(response)
    available = [item.get("name", "") for item in payload.get("models", [])]
    if not any(name == model or name == f"{model}:latest" or name.startswith(model + ":") for name in available):
        raise ConnectionError(f"Model {model!r} is not available through {base_url}; found: {available}")


def load_targets(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Target metadata must be a JSON array of objects")
    return records


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("filename") and record.get("classification", {}).get("classification"):
                    completed.add(record["filename"].lower())
            except (json.JSONDecodeError, AttributeError):
                continue
    return completed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_issue(path: Path, filename: str, source_path: Path | None, error: Exception) -> None:
    issue = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "source_path": source_path.as_posix() if source_path else None,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(issue, ensure_ascii=False) + "\n")
        handle.flush()


def process_pdf(pdf_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    pages = parse_pdf(pdf_path)
    document_text, source_pages = scale_representative_text(pages, args.max_chars)
    if not document_text:
        raise ValueError("PDF contains no extractable text")

    extraction = None
    for attempt in range(args.retries + 1):
        try:
            extraction = call_ollama(
                args.ollama_url, args.model, EXTRACTION_SYSTEM_PROMPT,
                "Analyze this scientific document. Extract factual scale evidence only.\n\n"
                "DOCUMENT TEXT:\n" + document_text,
                EXTRACTION_SCHEMA, args.timeout, args.context_length, args.num_predict,
            )
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            if attempt == args.retries:
                raise
            time.sleep(2**attempt)

    classifier = None
    extracted_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    for attempt in range(args.retries + 1):
        try:
            classifier = call_ollama(
                args.ollama_url, args.model, CLASSIFIER_SYSTEM_PROMPT,
                "Classify this extracted evidence. Do not read additional source material.\n\n"
                "EXTRACTED EVIDENCE JSON:\n" + extracted_json,
                CLASSIFIER_SCHEMA, args.timeout, args.context_length, 1024,
            )
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            if attempt == args.retries:
                raise
            time.sleep(2**attempt)

    classification = classifier.get("classification")
    allowed = {item for item in CLASSIFIER_SCHEMA["properties"]["classification"]["enum"]}
    if classification not in allowed:
        raise ValueError(f"Unexpected classifier label: {classification!r}")
    return {
        "filename": pdf_path.name,
        "source_path": pdf_path.resolve().as_posix(),
        "content_sha256": file_sha256(pdf_path),
        "model": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pages": source_pages,
        "scale_evidence": extraction,
        "classification": classifier,
        "tags": [classification, "SCALE_DERIVED_PROXY", "NOT_OFFICIAL_TRL"],
    }


def write_report(output_path: Path, report_path: Path, target_count: int, failures: int) -> None:
    records: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    counts = Counter(record.get("classification", {}).get("classification") for record in records)
    labels = ("TRL_2_3_PROXY", "TRL_4_5_PROXY", "TRL_6_7_PROXY", "INSUFFICIENT_EVIDENCE", "REVIEW_EXCLUDED")
    report = {
        "schema_version": "1.0",
        "classification_type": "scale_derived_trl_proxy",
        "official_trl_claimed": False,
        "target_records": target_count,
        "processed_records": len(records),
        "failed_records": failures,
        "coverage": {
            label: {
                "count": counts[label],
                "fraction_of_target": counts[label] / target_count if target_count else 0,
            }
            for label in labels
        },
        "records": records,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both TRL scale-evidence prompts over a PDF set.")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS, help="Metadata JSON array containing document.filename values")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "all_pdfs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Resumable tagged JSONL output")
    parser.add_argument("--report", type=Path, default=None, help="Coverage JSON path; defaults next to --output")
    parser.add_argument("--model", default="qwen3-next:80b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-chars", type=int, default=60000)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N pending target PDFs")
    args = parser.parse_args()

    targets = load_targets(args.targets)
    target_names = [record.get("document", {}).get("filename") for record in targets]
    if any(not name for name in target_names):
        raise ValueError("Every target record must contain document.filename")
    if len({name.lower() for name in target_names}) != len(target_names):
        raise ValueError("Target list contains duplicate PDF filenames")

    pdf_paths: dict[str, Path] = {}
    for path in args.pdf_dir.rglob("*.pdf"):
        pdf_paths.setdefault(path.name.lower(), path)
    missing = [name for name in target_names if name.lower() not in pdf_paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} target PDFs are missing; first examples: {missing[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_name(args.output.stem + "_coverage.json")
    issue_path = args.output.with_name(args.output.stem + "_issues.jsonl")
    completed = load_completed(args.output)
    pending = [pdf_paths[name.lower()] for name in target_names if name.lower() not in completed]
    if args.limit:
        pending = pending[:args.limit]
    print(f"targets={len(target_names)} completed={len(completed)} pending_this_run={len(pending)}", flush=True)

    if pending:
        check_ollama(args.ollama_url, args.model)
    for index, pdf_path in enumerate(pending, start=1):
        try:
            record = process_pdf(pdf_path, args)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"[{index}/{len(pending)}] {pdf_path.name}: {record['classification']['classification']}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted; completed JSONL records are preserved and the run can resume.", flush=True)
            break
        except Exception as error:
            write_issue(issue_path, pdf_path.name, pdf_path, error)
            print(f"[{index}/{len(pending)}] {pdf_path.name}: {type(error).__name__}: {error}", flush=True)

    failed = sum(1 for line in issue_path.read_text(encoding="utf-8").splitlines() if line.strip()) if issue_path.exists() else 0
    write_report(args.output, report_path, len(target_names), failed)
    print(f"tagged_jsonl={args.output}", flush=True)
    print(f"coverage_json={report_path}", flush=True)
    print(f"issues_jsonl={issue_path}", flush=True)


if __name__ == "__main__":
    main()