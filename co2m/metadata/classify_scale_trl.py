"""Classify Prompt 1 scale-evidence records and write tagged JSON coverage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REVIEW_TYPES = {"REVIEW", "PERSPECTIVE"}
NOT_PERFORMED = re.compile(
    r"\b(future work|will be (?:built|tested|operated)|planned|proposed|"
    r"previous study|prior study|cited study|literature reports?)\b",
    re.IGNORECASE,
)
HIGH_SCALE_TERMS = (
    "pilot plant", "demonstration plant", "demonstration system",
    "industrial slipstream", "field test", "field operation",
    "industrial facility", "commercial plant",
)
MID_SCALE_TERMS = (
    "bench-scale", "bench scale", "integrated skid", "pilot skid",
    "engineering-scale", "engineering scale", "prototype",
    "continuous reactor", "continuous-flow",
)
LOW_SCALE_TERMS = (
    "laboratory reactor", "lab-scale", "laboratory-scale", "laboratory scale",
    "small autoclave", "flask", "beaker", "vial", "tube reactor",
    "electrochemical cell", "coupon-scale", "coupon scale",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _active_evidence_text(entry: dict[str, Any]) -> str:
    context = _text(entry.get("context"))
    quote = _text(entry.get("evidence_quote"))
    if NOT_PERFORMED.search(f"{context} {quote}"):
        return ""
    return f"{context} {quote}".strip()


def _number(value: str) -> float | None:
    match = re.search(r"(?<![\w.])([\d,]+(?:\.\d+)?(?:e[+-]?\d+)?)", value, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _unit(text: str) -> str:
    normalized = text.lower().replace("μ", "u").replace("µ", "u")
    if re.search(r"\b(?:kg\s*/\s*(?:h|hr|hours?)|kg\s*per\s*hour)\b", normalized):
        return "kg_per_hour"
    if re.search(r"\b(?:kg\s*/\s*(?:d|days?)|kg\s*per\s*day)\b", normalized):
        return "kg_per_day"
    if re.search(r"\b(?:t\s*/\s*d(?:ay)?|tonnes?\s*per\s*day)\b", normalized):
        return "tonnes_per_day"
    if re.search(r"\b(?:tonnes?|metric tons?)\b|\bt\s*/", normalized):
        return "tonne"
    if re.search(r"\bkg\b", normalized):
        return "kg"
    if re.search(r"\b(?:mg|milligrams?)\b", normalized):
        return "mg"
    if re.search(r"\b(?:g|grams?)\b", normalized):
        return "g"
    if re.search(r"\b(?:m\s*3|m3|cubic meters?)\b", normalized):
        return "m3"
    if re.search(r"\b(?:ml|millilit(?:er|re)s?)\b", normalized):
        return "ml"
    if re.search(r"\b(?:l|lit(?:er|re)s?)\b", normalized):
        return "l"
    return ""


def _quantitative_levels(record: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    low = False
    medium = False
    evidence: list[str] = []
    for item in record.get("quantitative_scale_evidence", []):
        if not isinstance(item, dict) or item.get("is_primary_process_scale") is not True:
            continue
        active_text = _active_evidence_text(item)
        if not active_text:
            continue
        amount = _number(f"{_text(item.get('value'))} {_text(item.get('unit'))}")
        unit = _unit(f"{_text(item.get('value'))} {_text(item.get('unit'))} {_text(item.get('parameter'))}")
        parameter = _text(item.get("parameter")).lower()
        level = ""
        if amount is not None and unit in {"kg", "g", "mg", "tonne"}:
            mass_kg = amount * {"kg": 1, "g": 0.001, "mg": 0.000001, "tonne": 1000}[unit]
            if mass_kg >= 10:
                level = "high"
            elif mass_kg >= 0.1:
                level = "medium"
            else:
                level = "low"
        elif amount is not None and unit in {"l", "ml", "m3"}:
            volume_l = amount * {"l": 1, "ml": 0.001, "m3": 1000}[unit]
            if volume_l >= 100:
                level = "high"
            elif volume_l >= 1:
                level = "medium"
            else:
                level = "low"
        elif amount is not None and unit in {"kg_per_hour", "kg_per_day", "tonnes_per_day"}:
            rate_kg_day = amount * {"kg_per_hour": 24, "kg_per_day": 1, "tonnes_per_day": 1000}[unit]
            if unit in {"kg_per_hour", "tonnes_per_day"} or rate_kg_day >= 10:
                level = "high"
            elif rate_kg_day >= 1:
                level = "medium"
        if level:
            low = True
            medium |= level in {"medium", "high"}
            evidence.append(_text(item.get("evidence_quote")) or f"{parameter}: {item.get('value')} {item.get('unit')}")
            if level == "high":
                evidence.append("quantitative scale meets pilot/demo proxy guidance")
            elif level == "medium":
                evidence.append("quantitative scale meets bench/integrated proxy guidance")
    return low, medium, evidence


def _qualitative_text(record: dict[str, Any]) -> tuple[str, list[str]]:
    parts: list[str] = []
    quotes: list[str] = []
    for item in record.get("equipment_scale_evidence", []):
        if isinstance(item, dict):
            active_text = _active_evidence_text(item)
            if active_text:
                parts.append(" ".join((
                    _text(item.get("equipment")),
                    _text(item.get("size_or_capacity")),
                    active_text,
                )))
                quote = _text(item.get("evidence_quote"))
                if quote:
                    quotes.append(quote)
    largest = record.get("largest_demonstrated_scale", {})
    if isinstance(largest, dict):
        active_text = _active_evidence_text(largest)
        if active_text:
            parts.append(active_text)
            quote = _text(largest.get("evidence_quote"))
            if quote:
                quotes.append(quote)
    operating = record.get("operating_scale_evidence", {})
    if isinstance(operating, dict):
        parts.append(" ".join(_text(value) for value in operating.values()))
    terms = record.get("scale_terms_found", [])
    parts.extend(_text(term) for term in terms if term)
    return " ".join(parts).lower(), quotes


def classify(record: dict[str, Any]) -> dict[str, Any]:
    document_type = _text(record.get("document_type")).strip().upper()
    explicit_trl = record.get("explicit_trl")
    if not isinstance(explicit_trl, dict):
        explicit_trl = {"reported": False, "trl_value": None, "evidence": None}

    if record.get("exclude_from_scale_classification") is True or document_type in REVIEW_TYPES:
        label = "REVIEW_EXCLUDED"
        reasons = ["document is marked as a review/perspective or explicitly excluded"]
    elif document_type == "MODELING_ONLY":
        label = "INSUFFICIENT_EVIDENCE"
        reasons = ["modeled scale is not evidence of a process system demonstrated in the document"]
    elif record.get("evidence_sufficient_for_scale_classification") is False:
        label = "INSUFFICIENT_EVIDENCE"
        reasons = ["extractor marked the scale evidence insufficient"]
    else:
        has_primary, has_medium, quantitative_evidence = _quantitative_levels(record)
        qualitative, qualitative_quotes = _qualitative_text(record)
        high_terms = [term for term in HIGH_SCALE_TERMS if term in qualitative]
        medium_terms = [term for term in MID_SCALE_TERMS if term in qualitative]
        low_terms = [term for term in LOW_SCALE_TERMS if term in qualitative]

        if high_terms:
            label = "TRL_6_7_PROXY"
            reasons = [f"operated/tested scale evidence includes: {', '.join(high_terms)}"]
        elif has_medium and quantitative_evidence and any(
            "quantitative scale meets pilot/demo" in item for item in quantitative_evidence
        ):
            label = "TRL_6_7_PROXY"
            reasons = ["primary-process quantity meets pilot/demo proxy guidance"]
        elif medium_terms or has_medium:
            label = "TRL_4_5_PROXY"
            reasons = ["primary-process quantity or equipment indicates bench/integrated scale"]
        elif has_primary or low_terms:
            label = "TRL_2_3_PROXY"
            reasons = ["primary-process quantity or equipment indicates laboratory scale"]
        else:
            label = "INSUFFICIENT_EVIDENCE"
            reasons = ["no usable primary-process scale quantity or operated equipment scale was extracted"]
        reasons.extend(quantitative_evidence)
        reasons.extend(qualitative_quotes)

    tags = [label, "SCALE_DERIVED_PROXY", "NOT_OFFICIAL_TRL"]
    return {
        "classification": label,
        "tags": tags,
        "is_official_trl": False,
        "document_type": document_type or None,
        "explicit_trl": explicit_trl,
        "classification_reasons": reasons,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        payload = json.loads(content)
        values = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Each input record must be a JSON object")
    return values


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    tagged_records = []
    counts: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        identity = next((record.get(key) for key in ("document_id", "filename", "source_path", "doi") if record.get(key)), None)
        result = classify(record)
        counts[result["classification"]] += 1
        tagged_records.append({
            "record_id": identity or f"record_{index:06d}",
            "scale_derived_trl_proxy": result,
            "extracted_scale_evidence": record,
        })
    return {
        "schema_version": "1.0",
        "classification_type": "scale_derived_trl_proxy",
        "official_trl_claimed": False,
        "input_records": len(records),
        "coverage": {
            label: {"count": counts[label], "fraction": counts[label] / len(records) if records else 0}
            for label in (
                "TRL_2_3_PROXY", "TRL_4_5_PROXY", "TRL_6_7_PROXY",
                "INSUFFICIENT_EVIDENCE", "REVIEW_EXCLUDED",
            )
        },
        "records": tagged_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag Prompt 1 scale-extraction records with scale-derived TRL proxy bands."
    )
    parser.add_argument("--input", required=True, type=Path, help="Prompt 1 output (.json or .jsonl)")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("scale_trl_tags.json"),
        help="Tagged JSON report (default: co2m/metadata/scale_trl_tags.json)",
    )
    args = parser.parse_args()
    report = build_report(_load_records(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report['input_records']} tagged records to {args.output}")


if __name__ == "__main__":
    main()