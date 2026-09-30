#!/usr/bin/env python
"""Generate structured document metadata from PDFs with a local Ollama model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingest_postgres import parse_pdf


METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "authors": {"type": "array", "items": {"type": "string"}},
        "publication_year": {"type": ["integer", "null"]},
        "doi": {"type": ["string", "null"]},
        "document_type": {"type": ["string", "null"]},
        "summary": {"type": "string", "maxLength": 1200},
        "materials": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
        "material_classes": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "co2_processes": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "measurement_methods": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "reported_properties": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "scale_stage": {
            "type": ["string", "null"],
            "enum": ["laboratory", "bench", "pilot", "demonstration", "commercial", None],
        },
        "scale_stage_evidence": {"type": ["string", "null"]},
        "trl_source": {
            "type": "string",
            "enum": ["reported", "inferred", "not_assessed"],
        },
        "reported_trl": {"type": ["integer", "null"], "minimum": 1, "maximum": 9},
        "inferred_trl_min": {"type": ["integer", "null"], "minimum": 2, "maximum": 7},
        "inferred_trl_max": {"type": ["integer", "null"], "minimum": 2, "maximum": 7},
        "max_directly_demonstrated_trl": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 9,
        },
        "assessment_basis": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "evidence_refs": {"type": "array", "items": {"type": "string", "pattern": "^EVID_[0-9]{3}$"}, "maxItems": 8},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string", "pattern": "^EVID_[0-9]{3}$"},
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string", "maxLength": 300},
                },
                "required": ["evidence_id", "page", "quote"],
                "additionalProperties": False,
            },
            "maxItems": 8,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "title", "authors", "publication_year", "doi", "document_type",
        "summary", "materials", "material_classes", "co2_processes",
        "measurement_methods", "reported_properties", "keywords", "scale_stage",
        "scale_stage_evidence", "trl_source", "reported_trl", "inferred_trl_min",
        "inferred_trl_max", "max_directly_demonstrated_trl", "assessment_basis",
        "evidence_refs", "evidence", "confidence",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You extract metadata from scientific literature about CO2 capture,
conversion, storage, hydrogen, sustainable fuels and chemicals, batteries and energy
storage, and industrial decarbonization. Return only data supported by the supplied
PDF text. Do not guess missing bibliographic facts. Use null or an empty array when
evidence is absent. Keep names and measurement strings faithful to the source.
Classify scale_stage as exactly one of laboratory, bench, pilot, demonstration,
or commercial. Assess TRL using these bands: laboratory 2-3, bench 3-4, pilot 4-5,
demonstration 5-6, and commercial 6-7. Base classifications on the technology
actually studied, not scale words in citations or search metadata. Assess every
domain-relevant document: use reported when the source explicitly states a TRL,
inferred when the documented operating scale supports a band, and not_assessed only
when the supplied text is insufficient. Never assign a higher stage than the source
directly supports.

Set trl_source to reported only when the supplied text explicitly reports a TRL and
put that integer in reported_trl. Otherwise use inferred and populate inferred_trl_min,
inferred_trl_max, max_directly_demonstrated_trl, and concise snake_case
assessment_basis codes such as pilot_scale_operation, relevant_feed, or
integrated_subsystem. Use not_assessed and null numeric fields for unrelated papers
or insufficient evidence, and explain the reason in assessment_basis when possible.
Do not mark a record as not_assessed merely because the source does not use the word
TRL if its documented scale and operating evidence support an inference.

Create evidence entries with sequential IDs such as EVID_001. Every assessed scale,
TRL, transition, quantitative fact, challenge, outcome, or deployment claim must have
at least one supporting evidence entry. Each entry must contain the source page and a
short verbatim quote. evidence_refs must contain only IDs present in evidence and
relevant to the TRL assessment. Set scale_stage_evidence to one of those IDs, or null
when scale is not assessed. Confidence is an overall score from 0 to 1 for the
extracted record."""


def representative_text(pages: list[tuple[int, str]], max_chars: int) -> tuple[str, list[int]]:
    """Select front matter plus relevant pages while preserving page provenance."""
    selected: list[tuple[int, str]] = []
    keywords = (
        "co2", "carbon dioxide", "capture", "adsorp", "absorp", "mineral", "sorbent",
        "hydrogen", "electrolysis", "electrolyzer", "methane pyrolysis", "reforming",
        "syngas", "biomass", "biofuel", "sustainable aviation", "e-fuel", "methanol",
        "ammonia", "fischer-tropsch", "battery", "batteries", "electrode",
        "lithium-ion", "energy storage", "process heat", "waste heat", "heat recovery",
        "process intensification", "pilot", "demonstration", "commercial", "scale-up",
        "scaleup", "throughput", "capacity", "operating duration", "technology readiness",
        "trl", "facility", "plant", "deployment", "integrated system",
    )
    for page_number, text in pages:
        if page_number <= 3 or any(keyword in text.lower() for keyword in keywords):
            selected.append((page_number, text))

    parts: list[str] = []
    used_pages: list[int] = []
    size = 0
    for page_number, text in selected:
        section = f"\n--- PAGE {page_number} ---\n{text}"
        remaining = max_chars - size
        if remaining <= 0:
            break
        parts.append(section[:remaining])
        used_pages.append(page_number)
        size += min(len(section), remaining)
    return "".join(parts), used_pages


import re

def parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON output from LLM, handling markdown code fences, control chars, and truncation."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError:
        pass

    cleaned = "".join(ch if ord(ch) >= 32 or ch in "\n\r\t" else f"\\u{ord(ch):04x}" for ch in content)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # Delimiter-stack based JSON repair for truncated outputs
    stack = []
    in_string = False
    escape = False

    for ch in cleaned:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if stack:
                    stack.pop()

    repaired = cleaned.rstrip()
    if in_string:
        repaired += '"'
    
    repaired = re.sub(r'[,:\s]+$', '', repaired)

    while stack:
        opener = stack.pop()
        if opener == '{':
            repaired += '}'
        elif opener == '[':
            repaired += ']'

    try:
        return json.loads(repaired, strict=False)
    except json.JSONDecodeError:
        last_comma = max(repaired.rfind(','), repaired.rfind('{'), repaired.rfind('['))
        if last_comma > 0:
            truncated_base = repaired[:last_comma].rstrip()
            truncated_base = re.sub(r'[,:\s]+$', '', truncated_base)
            
            stack = []
            in_str = False
            esc = False
            for ch in truncated_base:
                if esc:
                    esc = False
                    continue
                if ch == '\\' and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if ch in '{[':
                        stack.append(ch)
                    elif ch in '}]':
                        if stack:
                            stack.pop()
            if in_str:
                truncated_base += '"'
            while stack:
                op = stack.pop()
                truncated_base += '}' if op == '{' else ']'
            return json.loads(truncated_base, strict=False)
        raise


def ollama_extract(
    base_url: str,
    model: str,
    text: str,
    timeout: int,
    context_length: int,
    num_predict: int = 8192,
) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "stream": False,
        "think": False,
        "format": METADATA_SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": context_length,
            "num_predict": num_predict,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Extract metadata from this PDF text:\n" + text},
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
    metadata = parse_json_response(payload["message"]["content"])
    normalize_trl_assessment(metadata)
    validate_trl_assessment(metadata)
    return metadata


def normalize_trl_assessment(metadata: dict[str, Any]) -> None:
    evidence_ids = {
        item["evidence_id"]
        for item in (metadata.get("evidence") or [])
        if isinstance(item, dict) and "evidence_id" in item
    }
    if metadata.get("evidence_refs"):
        metadata["evidence_refs"] = [
            ref for ref in metadata["evidence_refs"] if ref in evidence_ids
        ]
    if metadata.get("scale_stage_evidence") not in evidence_ids:
        metadata["scale_stage_evidence"] = None

    # Preserve assessments for all supported energy domains, not only records
    # that use the legacy co2_processes field.
    has_domain_assessment = any(
        metadata.get(field)
        for field in (
            "co2_processes",
            "materials",
            "measurement_methods",
            "reported_properties",
            "keywords",
            "scale_stage",
            "evidence_refs",
        )
    )
    if has_domain_assessment:
        if metadata.get("trl_source") == "inferred":
            demonstrated = metadata.get("max_directly_demonstrated_trl")
            if demonstrated is None:
                demonstrated = {
                    "laboratory": 3,
                    "bench": 4,
                    "pilot": 5,
                    "demonstration": 6,
                    "commercial": 7,
                }.get(metadata.get("scale_stage"))
                metadata["max_directly_demonstrated_trl"] = demonstrated
            if demonstrated is None:
                metadata.update({
                    "scale_stage": None,
                    "scale_stage_evidence": None,
                    "trl_source": "not_assessed",
                    "reported_trl": None,
                    "inferred_trl_min": None,
                    "inferred_trl_max": None,
                    "max_directly_demonstrated_trl": None,
                    "assessment_basis": [],
                    "evidence_refs": [],
                })
                return
            bands = {
                2: (2, 3, "laboratory"),
                3: (2, 3, "laboratory"),
                4: (3, 4, "bench"),
                5: (4, 5, "pilot"),
                6: (5, 6, "demonstration"),
                7: (6, 7, "commercial"),
            }
            if demonstrated in bands:
                inferred_min, inferred_max, scale_stage = bands[demonstrated]
                metadata["inferred_trl_min"] = inferred_min
                metadata["inferred_trl_max"] = inferred_max
                metadata["scale_stage"] = scale_stage
        return
    metadata.update({
        "scale_stage": None,
        "scale_stage_evidence": None,
        "trl_source": "not_assessed",
        "reported_trl": None,
        "inferred_trl_min": None,
        "inferred_trl_max": None,
        "max_directly_demonstrated_trl": None,
        "assessment_basis": [],
        "evidence_refs": [],
    })


def validate_trl_assessment(metadata: dict[str, Any]) -> None:
    source = metadata["trl_source"]
    reported = metadata["reported_trl"]
    inferred_min = metadata["inferred_trl_min"]
    inferred_max = metadata["inferred_trl_max"]
    demonstrated = metadata["max_directly_demonstrated_trl"]

    if source == "reported" and reported is None:
        raise ValueError("reported TRL requires reported_trl")
    if source == "inferred":
        if inferred_min is None or inferred_max is None or demonstrated is None:
            raise ValueError("inferred TRL requires bounds and demonstrated TRL")
        if inferred_min > inferred_max or not inferred_min <= demonstrated <= inferred_max:
            raise ValueError("inferred TRL bounds are inconsistent")
        if (inferred_min, inferred_max) not in {(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}:
            raise ValueError("inferred TRL must use an allowed adjacent range")
    if source == "not_assessed" and any(
        value is not None for value in (reported, inferred_min, inferred_max, demonstrated)
    ):
        raise ValueError("not_assessed TRL must have null numeric values")

    evidence_ids = {item["evidence_id"] for item in metadata["evidence"]}
    references = set(metadata["evidence_refs"])
    scale_reference = metadata.get("scale_stage_evidence")
    if not references <= evidence_ids:
        raise ValueError("evidence_refs contains an unknown evidence ID")
    if scale_reference is not None and scale_reference not in evidence_ids:
        raise ValueError("scale_stage_evidence contains an unknown evidence ID")


def load_completed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(json.loads(line)["filename"].lower())
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def deduplicate_output(output_path: Path) -> tuple[int, int]:
    """Atomically keep the first valid record for each PDF filename."""
    if not output_path.exists():
        return 0, 0
    records = []
    filenames = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            filename = record["filename"].lower()
            if filename not in filenames:
                records.append(record)
                filenames.add(filename)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_path.write_text(temporary_path.read_text(encoding="utf-8"), encoding="utf-8")
    temporary_path.unlink()
    return sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line), len(records)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", type=Path, default=Path("all_pdfs"))
    parser.add_argument("--output", type=Path, default=Path("co2m/metadata/qwen_metadata.jsonl"))
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-chars", type=int, default=60000)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--num-predict", type=int, default=4096, help="Max output tokens for the model response; increase if JSON responses are truncated.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deduplicate-only", action="store_true")
    args = parser.parse_args()

    if args.deduplicate_only:
        before = sum(1 for line in args.output.read_text(encoding="utf-8").splitlines() if line)
        _, after = deduplicate_output(args.output)
        print(f"before={before} after={after}")
        return

    pdfs = sorted(args.pdf_dir.rglob("*.pdf"))
    if args.limit:
        pdfs = pdfs[:args.limit]
    completed = set() if args.force else load_completed(args.output)
    pending = [pdf for pdf in pdfs if pdf.name.lower() not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"pdfs={len(pdfs)} completed={len(completed)} pending={len(pending)}")
    output_mode = "w" if args.force else "a"
    cause_of_ending = "completed"
    if pending:
        check_ollama_server(args.ollama_url, args.model)
    with args.output.open(output_mode, encoding="utf-8") as output:
        for index, pdf_path in enumerate(pending, 1):
            try:
                pages = parse_pdf(pdf_path)
                text, used_pages = representative_text(pages, args.max_chars)
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
                        time.sleep(2 ** attempt)
                record = {
                    "filename": pdf_path.name,
                    "source_path": str(pdf_path.as_posix()),
                    "content_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                    "model": args.model,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_pages": used_pages,
                    "metadata": metadata,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{index}/{len(pending)}] {pdf_path.name}: ok", flush=True)
            except KeyboardInterrupt:
                cause_of_ending = "interrupted_by_user"
                print(f"\n[{index}/{len(pending)}] Run interrupted by user (KeyboardInterrupt).", flush=True)
                break
            except Exception as error:
                print(f"[{index}/{len(pending)}] {pdf_path.name}: {type(error).__name__}: {error}", flush=True)

    print(f"cause_of_ending={cause_of_ending}")


if __name__ == "__main__":
    main()