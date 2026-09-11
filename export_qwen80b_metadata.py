"""Export Qwen 80B JSONL metadata as Sloan scale-up schema records."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "co2m" / "metadata" / "qwen3_next_80b_metadata.jsonl"
OUTPUT_PATH = ROOT / "co2m" / "metadata" / "qwen80b_10_pdfs_metadata.json"


def evidence_registry(evidence):
    if not isinstance(evidence, list):
        return []
    return [
        {
            "evidence_id": (item.get("evidence_id") or f"EVID_{index:03d}") if isinstance(item, dict) else f"EVID_{index:03d}",
            "page": item.get("page") if isinstance(item, dict) else None,
            "section": None,
            "chunk_id": None,
            "source_element": "text",
            "evidence_type": "claim",
            "evidence_subject_role": "technology_assessment",
            "evidence_text": (item.get("quote") or "") if isinstance(item, dict) else "",
            "extraction_status": "accepted",
        }
        for index, item in enumerate(evidence, start=1)
    ]


def convert_record(record, index):
    metadata = record.get("metadata") or {}
    processes = [p for p in (metadata.get("co2_processes") or []) if isinstance(p, str)]
    keywords = [k for k in (metadata.get("keywords") or []) if isinstance(k, str)]
    materials = [m for m in (metadata.get("materials") or []) if isinstance(m, str)]
    evidence = [e for e in (metadata.get("evidence") or []) if isinstance(e, dict)]
    evidence_refs = [r for r in (metadata.get("evidence_refs") or []) if isinstance(r, str)]
    scale_stage = metadata.get("scale_stage")
    trl_source = metadata.get("trl_source") or "not_assessed"
    title = metadata.get("title") or ""
    technology_name = processes[0] if processes else (title or record.get("filename") or "")
    energy_relevant = bool(processes or any("co2" in (keyword or "").lower() for keyword in keywords))
    scaleup_relevant = bool(scale_stage or evidence_refs)

    return {
        "schema_version": "sloan_scaleup_metadata_v1.0",
        "document": {
            "document_id": f"DOC_{index:06d}",
            "filename": record.get("filename") or "",
            "content_sha256": record.get("content_sha256") or "",
            "title": title,
            "authors": metadata.get("authors") or [],
            "publication_year": metadata.get("publication_year"),
            "publication_date": None,
            "document_type": metadata.get("document_type") or "",
            "journal_or_source": "",
            "publisher": "",
            "doi": metadata.get("doi"),
            "report_number": None,
            "patent_number": None,
            "discovery_source": "qwen3-next:80b",
            "source_url": None,
            "language": "en",
            "duplicate_cluster_id": None,
            "version_type": None,
        },
        "screening": {
            "energy_domain_relevant": energy_relevant,
            "scaleup_relevant": scaleup_relevant,
            "inclusion_decision": "include" if energy_relevant else "exclude",
            "relevance_evidence": processes + keywords,
            "explicit_scale_stage_present": scale_stage is not None,
            "quantitative_scale_indicator_present": bool(
                metadata.get("reported_properties") or metadata.get("measurement_methods")
            ),
            "explicit_trl_present": trl_source == "reported",
            "scaleup_transition_present": False,
            "selection_mode": "automatic",
            "borderline": False,
            "exclusion_reason": None if energy_relevant else "Not relevant to CO2 process scale-up",
        },
        "taxonomy": {
            "taxonomy_version": "process_scaleup_topics_v1",
            "discovery_topics": keywords,
            "primary_topic": {
                "core_id": "CO2_CAPTURE" if any("capture" in (item or "").lower() for item in processes) else "",
                "core_technology": technology_name,
                "sub_id": "",
                "sub_technology": "",
            },
            "secondary_topics": [],
            "variant_tags": [],
            "cross_cutting_tags": [],
            "application_sectors": [],
        },
        "study": {
            "study_types": ["experimental"] if scale_stage else [],
            "primary_evidence_role": "technology_assessment" if evidence else "",
            "study_scope": title,
            "experimental": bool(scale_stage) if scale_stage is not None else None,
            "modeling_only": None,
            "review_article": None,
            "technoeconomic_analysis": "techno-economic" in title.lower() if title else False,
            "life_cycle_analysis": "life cycle" in title.lower() if title else False,
        },
        "technology_entities": [{
            "technology_id": "TECH_0001",
            "canonical_name": technology_name,
            "aliases": [],
            "technology_family": "",
            "technology_route": "",
            "developer_or_owner": None,
            "project_name": None,
            "facility_name": None,
            "system_scope": "",
            "entity_status": "",
        }],
        "scale_observations": ([{
            "scale_id": "SCALE_001",
            "technology_id": "TECH_0001",
            "evidence_role": "technology_assessment",
            "scale_stage_normalized": scale_stage,
            "scale_stage_raw": scale_stage,
            "system_scope": "",
            "environment": "",
            "operating_mode": "",
            "directly_operated_in_this_study": True,
            "scale_description": title,
            "evidence_refs": evidence_refs,
        }] if scale_stage else []),
        "trl_assessments": ([{
            "trl_id": "TRL_001",
            "technology_id": "TECH_0001",
            "trl_source": trl_source,
            "reported_trl": metadata.get("reported_trl"),
            "inferred_trl_min": metadata.get("inferred_trl_min"),
            "inferred_trl_max": metadata.get("inferred_trl_max"),
            "max_directly_demonstrated_trl": metadata.get("max_directly_demonstrated_trl"),
            "assessment_basis": metadata.get("assessment_basis", []),
            "assessment_confidence": str(metadata.get("confidence", "")),
            "current_trl": metadata.get("max_directly_demonstrated_trl"),
            "target_trl": None,
            "target_trl_is_future": None,
            "evidence_refs": evidence_refs,
        }] if trl_source != "not_assessed" else []),
        "process_system": {
            "process_description": metadata.get("summary", ""),
            "unit_operations": [{
                "sequence": position,
                "operation": operation,
                "equipment_type": "",
                "technology_id": "TECH_0001",
            } for position, operation in enumerate(processes, start=1)],
            "equipment": [],
            "substances": [{
                "name": material,
                "normalized_name": material,
                "role": "process material",
            } for material in materials],
            "process_configuration": processes,
        },
        "quantitative_facts": [],
        "scale_up_transitions": [],
        "scaleup_challenges": [],
        "scale_up_outcomes": [],
        "safety_and_materials": {
            "materials_of_construction": [],
            "compatibility_issues": [],
            "hazards": [],
            "safeguards": [],
        },
        "deployment": {
            "project_name": None,
            "facility_name": None,
            "organization": None,
            "location": {"country": None, "state_or_region": None, "site": None},
            "project_status": None,
            "commissioning_year": None,
            "operation_start_year": None,
            "operation_end_year": None,
            "reported_operating_duration": None,
        },
        "evidence_registry": evidence_registry(evidence),
        "document_summary": {
            "technical_summary": metadata.get("summary", ""),
            "scaleup_summary": "Scale-stage evidence extracted from the source document." if scale_stage else "No scale-stage evidence extracted.",
            "key_scaleup_lessons": [],
        },
        "quality_control": {
            "parse_quality": "good",
            "extraction_warnings": [],
            "validation_status": "accepted",
            "needs_review": False,
            "review_reasons": [],
        },
        "extraction_provenance": {
            "model": record.get("model", "qwen3-next:80b"),
            "model_version": "80B",
            "prompt_version": "sloan_scaleup_metadata_v1.0",
            "extraction_timestamp": record.get("generated_at", ""),
            "pipeline_version": "qwen3-next:80b",
        },
    }


def main():
    source_records = [
        json.loads(line)
        for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [convert_record(record, index) for index, record in enumerate(source_records[:10], start=1)]
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert len(loaded) == 10
    assert all(item["extraction_provenance"]["model"] == "qwen3-next:80b" for item in loaded)
    print(f"Created {OUTPUT_PATH} with {len(loaded)} records.")


if __name__ == "__main__":
    main()