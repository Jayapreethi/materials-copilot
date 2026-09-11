#!/usr/bin/env python
"""
Diagnostic script to check vector database state and metadata extraction.
Helps identify why metadata coverage is 0% or collection size is doubled.
"""

import json
from pathlib import Path
from vector_db.vector_store import VectorStore
from metadata_enricher import MetadataEnricher

def diagnose_database():
    """Check database state and metadata."""
    
    print("=" * 80)
    print("VECTOR DATABASE DIAGNOSTIC")
    print("=" * 80)
    
    # Check if database exists
    db_path = Path("./vector_db/data")
    if not db_path.exists():
        print("\n❌ Database directory not found at:", db_path.absolute())
        return
    
    print(f"\n✓ Database directory found: {db_path.absolute()}")
    
    # Connect to database
    print("\n[1] Connecting to database...")
    store = VectorStore()
    
    # Get collection stats
    print("\n[2] Collection Statistics:")
    collection_count = store.collection.count()
    print(f"  Total documents: {collection_count}")
    
    if collection_count == 0:
        print("  ❌ Database is empty!")
        return
    
    # Get sample documents
    print("\n[3] Sampling database documents...")
    sample_size = 5
    results = store.collection.get(limit=sample_size, include=["metadatas", "documents"])
    
    print(f"\n  Sample of first {sample_size} documents:")
    for doc_id, metadata, text in zip(results["ids"], results["metadatas"], results["documents"]):
        print(f"\n  ID: {doc_id}")
        print(f"  Text: {text[:80]}...")
        print(f"  Metadata fields: {len(metadata)} fields")
        print(f"  Fields: {list(metadata.keys())[:10]}")
        if len(metadata) > 10:
            print(f"    ... and {len(metadata) - 10} more")
    
    # Analyze metadata
    print("\n[4] Metadata Analysis:")
    all_docs = store.collection.get(limit=10000, include=["metadatas"])
    metadatas = all_docs["metadatas"]
    
    # Check which fields are populated
    field_counts = {}
    for meta in metadatas:
        for key in meta.keys():
            field_counts[key] = field_counts.get(key, 0) + 1
    
    print(f"\n  Field Coverage (out of {len(metadatas)} documents):")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        percentage = 100 * count / len(metadatas)
        print(f"    {field}: {count} ({percentage:.1f}%)")
    
    # Check for specific chemistry fields
    print("\n[5] Chemistry Metadata Coverage:")
    chemistry_fields = [
        "material_class", "material_name", "co2_uptake_mmol_g",
        "temperature_K", "pressure_bar", "surface_area_m2_g",
        "measurement_method", "source_type"
    ]
    
    for field in chemistry_fields:
        count = sum(1 for m in metadatas if field in m)
        percentage = 100 * count / len(metadatas)
        status = "✓" if percentage > 0 else "✗"
        print(f"  {status} {field}: {count} ({percentage:.1f}%)")
    
    # Test enricher
    print("\n[6] Testing Metadata Enricher:")
    enricher = MetadataEnricher()
    
    test_text = """
    The metal-organic framework UiO-66-NH2 (Zr6O4(OH)4(BDC-NH2)6) was synthesized.
    The material exhibited a BET surface area of 1127 m2/g and pore volume of 0.48 cm3/g.
    Experimental CO2 adsorption measurements were conducted using volumetric adsorption 
    at 298 K and 1.0 bar. The CO2 uptake was 4.2 mmol/g with selectivity of 15.5.
    The heat of adsorption was 35 kJ/mol. Replicate measurements (n=3) showed uncertainty of ±5%.
    """
    
    enriched = enricher.enrich(test_text, doc_id=0, chunk_id=0, filename="test.pdf", page_number=1)
    
    print(f"\n  Enricher output for test text:")
    print(f"    Total fields: {len(enriched)}")
    
    extracted_fields = {k: v for k, v in enriched.items() if v is not None}
    print(f"    Non-null fields: {len(extracted_fields)}")
    
    if extracted_fields:
        print(f"\n    Extracted metadata:")
        for key, value in sorted(extracted_fields.items()):
            print(f"      ✓ {key}: {value}")
    else:
        print(f"\n    ❌ NO METADATA EXTRACTED!")
        print(f"    All fields returned as None")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if collection_count > 3700:
        if collection_count == 7378:
            print("\n⚠️  Collection size is 7378 (exactly 2 × 3689)")
            print("  This suggests ingestion ran twice. Options:")
            print("    1. Delete and rebuild: rm -rf ./vector_db/data/")
            print("    2. Or accept the doubled data (it won't affect queries)")
    
    if len(extracted_fields) == 0:
        print("\n⚠️  Enricher is not extracting metadata")
        print("  Possible causes:")
        print("    1. Metadata enricher patterns not matching text")
        print("    2. Enricher being called but output ignored")
        print("    3. Metadata being cleaned away before storage")
        print("\n  To fix:")
        print("    1. Check if _clean_metadata() is too aggressive")
        print("    2. Verify enricher.enrich() is being called")
        print("    3. Check if metadata is null before cleaning")
    
    if sum(1 for m in metadatas if "material_class" in m) == 0:
        print("\n⚠️  No material information extracted")
        print("  This could mean:")
        print("    1. PDF text doesn't contain material names")
        print("    2. Enricher patterns need adjustment")
        print("    3. Check sample documents to verify content")


if __name__ == "__main__":
    diagnose_database()
