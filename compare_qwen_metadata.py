#!/usr/bin/env python
"""Compare two Qwen-generated metadata JSONL files by PDF filename."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LIST_FIELDS = (
    "materials",
    "material_classes",
    "co2_processes",
    "measurement_methods",
    "reported_properties",
    "keywords",
    "assessment_basis",
    "evidence_refs",
)


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["filename"].lower()] = record
    return records


def comparable_trl(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
        metadata.get("trl_source"),
        metadata.get("reported_trl"),
        metadata.get("inferred_trl_min"),
        metadata.get("inferred_trl_max"),
        metadata.get("max_directly_demonstrated_trl"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("co2m/metadata/qwen_model_comparison.json"))
    args = parser.parse_args()

    baseline = load_records(args.baseline)
    candidate = load_records(args.candidate)
    filenames = sorted(baseline.keys() & candidate.keys())
    comparisons = []
    for filename in filenames:
        base_metadata = baseline[filename]["metadata"]
        candidate_metadata = candidate[filename]["metadata"]
        comparisons.append({
            "filename": filename,
            "scale_stage_8b": base_metadata.get("scale_stage"),
            "scale_stage_80b": candidate_metadata.get("scale_stage"),
            "scale_stage_agrees": base_metadata.get("scale_stage") == candidate_metadata.get("scale_stage"),
            "trl_8b": comparable_trl(base_metadata),
            "trl_80b": comparable_trl(candidate_metadata),
            "trl_agrees": comparable_trl(base_metadata) == comparable_trl(candidate_metadata),
            "confidence_8b": base_metadata.get("confidence"),
            "confidence_80b": candidate_metadata.get("confidence"),
            "evidence_count_8b": len(base_metadata.get("evidence", [])),
            "evidence_count_80b": len(candidate_metadata.get("evidence", [])),
            "field_counts": {
                field: {
                    "8b": len(base_metadata.get(field, [])),
                    "80b": len(candidate_metadata.get(field, [])),
                }
                for field in LIST_FIELDS
            },
        })

    report = {
        "baseline_model": next(iter(baseline.values())).get("model") if baseline else None,
        "candidate_model": next(iter(candidate.values())).get("model") if candidate else None,
        "compared_pdfs": len(filenames),
        "scale_stage_agreement": sum(item["scale_stage_agrees"] for item in comparisons),
        "trl_agreement": sum(item["trl_agrees"] for item in comparisons),
        "records": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"compared={len(filenames)} scale_agreement={report['scale_stage_agreement']} trl_agreement={report['trl_agreement']}")
    print(args.output)


if __name__ == "__main__":
    main()