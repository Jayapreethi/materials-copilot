#!/usr/bin/env python
"""
Test RAG pipeline to verify semantic search is working correctly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def test_vector_store():
    """Test vector store connection and retrieval."""
    print("\n" + "=" * 70)
    print("TEST 1: Vector Store Connection")
    print("=" * 70)
    
    from vector_db.vector_store import VectorStore
    
    store = VectorStore(persist_directory=str(PROJECT_ROOT / "vector_db" / "data"))
    count = store.collection.count()
    print(f"✓ Connected to Chroma collection")
    print(f"✓ Total documents in store: {count:,}")
    
    if count == 0:
        print("✗ ERROR: Vector database is empty!")
        return False
    
    stats = store.get_stats()
    print(f"✓ Collection name: {stats['collection_name']}")
    print(f"✓ Distance metric: {stats['distance_metric']}")
    print(f"✓ Persist directory: {stats['persist_directory']}")
    
    return True


def test_embeddings():
    """Test embedding generation."""
    print("\n" + "=" * 70)
    print("TEST 2: Embedding Generation")
    print("=" * 70)
    
    from embeddings.embeddings import EmbeddingGenerator
    
    generator = EmbeddingGenerator(device="cpu")
    print(f"✓ Embedding model loaded: {generator.model_name}")
    print(f"✓ Embedding dimension: {generator.dimension}")
    
    # Test embedding a sample query
    sample_query = "CO2 capture materials and sorbents"
    embedding = generator.embed_single(sample_query)
    print(f"✓ Generated embedding for: '{sample_query}'")
    print(f"✓ Embedding length: {len(embedding)}")
    
    if len(embedding) != generator.dimension:
        print(f"✗ ERROR: Expected dimension {generator.dimension}, got {len(embedding)}")
        return False
    
    return True


def test_semantic_search():
    """Test semantic search queries."""
    print("\n" + "=" * 70)
    print("TEST 3: Semantic Search")
    print("=" * 70)
    
    from rag_pipeline.query_service import RAGQueryService
    
    service = RAGQueryService(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path=str(PROJECT_ROOT / "vector_db" / "data"),
        collection_name="co2m_corpus",
        device="cpu",
    )
    
    # Test multiple queries
    test_queries = [
        "CO2 capture materials",
        "amine-based sorbents",
        "separation efficiency",
        "carbon dioxide absorption",
        "adsorption kinetics",
    ]
    
    print(f"\nTesting {len(test_queries)} semantic search queries:\n")
    
    all_ok = True
    for query in test_queries:
        try:
            results = service.query(
                question=query,
                top_k=3,
                include_context=False,
            )
            
            num_results = results.get("num_results", 0)
            if num_results > 0:
                top_result = results["results"][0]
                similarity = top_result.get("similarity", 0)
                filename = top_result.get("filename", "unknown")
                print(f"✓ Query: '{query}'")
                print(f"  → Found {num_results} results")
                print(f"  → Top match: {filename} (similarity: {similarity:.1%})")
            else:
                print(f"✗ Query: '{query}' - NO RESULTS")
                all_ok = False
        except Exception as e:
            print(f"✗ Query: '{query}' - ERROR: {e}")
            all_ok = False
        print()
    
    return all_ok


def test_rag_with_context():
    """Test RAG with context generation."""
    print("=" * 70)
    print("TEST 4: RAG with Context Generation")
    print("=" * 70)
    
    from rag_pipeline.query_service import RAGQueryService
    
    service = RAGQueryService(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path=str(PROJECT_ROOT / "vector_db" / "data"),
        collection_name="co2m_corpus",
        device="cpu",
    )
    
    query = "What are the most effective materials for CO2 capture?"
    print(f"\nQuery: '{query}'")
    print()
    
    try:
        results = service.query(
            question=query,
            top_k=5,
            include_context=True,
        )
        
        print(f"Found {results.get('num_results', 0)} relevant documents")
        print()
        
        for i, result in enumerate(results.get("results", []), 1):
            print(f"Result {i}:")
            print(f"  File: {result.get('filename', 'unknown')}")
            print(f"  Similarity: {result.get('similarity', 0):.1%}")
            print(f"  Page: {result.get('page_number', 'N/A')}")
            
            # Show a snippet of the context if available
            context = result.get("context", "")
            if context:
                snippet = context[:150].replace("\n", " ")
                print(f"  Context: {snippet}...")
            print()
        
        return results.get('num_results', 0) > 0
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retriever():
    """Test the RAG retriever directly."""
    print("=" * 70)
    print("TEST 5: RAG Retriever")
    print("=" * 70)
    
    from rag_pipeline.retriever import RAGRetriever
    
    retriever = RAGRetriever(
        vector_store_path=str(PROJECT_ROOT / "vector_db" / "data"),
    )
    
    print(f"✓ Retriever initialized")
    print(f"✓ Retriever stats: {retriever.get_stats()}")
    
    # Test batch retrieval
    batch_queries = [
        "CO2 adsorption",
        "material properties",
        "temperature effects",
    ]
    
    print(f"\nTesting batch retrieval with {len(batch_queries)} queries:")
    try:
        batch_results = retriever.batch_retrieve(
            queries=batch_queries,
            top_k=3,
        )
        
        print(f"✓ Batch retrieval completed")
        print(f"✓ Results for {len(batch_results)} queries:")
        
        for query, results in zip(batch_queries, batch_results):
            num_docs = len(results.get("documents", []))
            print(f"  - '{query}': {num_docs} documents")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all RAG tests."""
    print("\n" + "=" * 70)
    print("CO2M RAG SYSTEM VERIFICATION")
    print("=" * 70)
    
    tests = [
        ("Vector Store", test_vector_store),
        ("Embeddings", test_embeddings),
        ("Semantic Search", test_semantic_search),
        ("RAG with Context", test_rag_with_context),
        ("Retriever", test_retriever),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✓ PASS" if success else "✗ FAIL"
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "✗ CRASH"
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        print(f"{result} - {test_name}")
    
    all_passed = all("PASS" in result for result in results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - RAG System is fully operational!")
    else:
        print("✗ SOME TESTS FAILED - Check output above for details")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
