#!/usr/bin/env python
"""
example_rag_pipeline.py
Demonstration of the complete RAG pipeline for CO2M corpus.
"""

import logging
from rag_pipeline import RAGQueryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run RAG pipeline example."""

    # Initialize the query service
    print("=" * 70)
    print("CO2M RAG Pipeline Example")
    print("=" * 70)
    
    try:
        print("\n1️⃣ Initializing RAG Query Service...")
        service = RAGQueryService(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_path="./vector_db/data",
            collection_name="co2m_corpus",
            device="cpu",
        )
        
        stats = service.get_stats()
        print(f"✓ Service initialized")
        print(f"  - Vector Store: {stats['collection_name']}")
        print(f"  - Documents: {stats['total_documents']:,}")
        print(f"  - Embedding Model: {stats['embedding_model']}")
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return

    # Example queries
    example_queries = [
        "What materials are used for CO2 capture?",
        "How effective are amine-based sorbents?",
        "What is the role of metal-organic frameworks in CO2 separation?",
        "What are the latest advances in solid sorbent materials?",
    ]

    for i, query in enumerate(example_queries, 1):
        print(f"\n2️⃣ Query {i}: {query}")
        print("-" * 70)

        try:
            # Execute query
            results = service.query(
                question=query,
                top_k=3,
                concept_filter=None,
                include_context=True,
            )

            # Display results
            print(f"\n✓ Results: {results['num_results']} documents found\n")

            for result in results["results"]:
                print(f"  [{result['rank']}] {result['filename']} | Page {result['page_number']}")
                print(f"      Similarity: {result['similarity']:.3f}")
                print(f"      {result['text'][:150]}...")
                if result.get("material_name"):
                    print(f"      Material: {result['material_name']}")
                if result.get("co2_uptake_mmol_g"):
                    print(f"      CO₂ Uptake: {result['co2_uptake_mmol_g']} mmol/g")
                print()

            # Show summary
            if "context_summary" in results:
                summary = results["context_summary"]
                print(f"  Summary:")
                print(f"    - Avg Similarity: {summary['avg_similarity']:.3f}")
                print(f"    - Materials Found: {summary['materials_found']}")
                if "co2_uptake_stats" in summary:
                    stats = summary["co2_uptake_stats"]
                    print(f"    - CO₂ Uptake Avg: {stats['avg']:.2f} mmol/g")

        except Exception as e:
            print(f"✗ Query failed: {e}")
            continue

    print("\n" + "=" * 70)
    print("RAG Pipeline Example Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
