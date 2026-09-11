#!/usr/bin/env python
"""
Interactive database query tool.
Usage: python query_database.py
"""

import json
from vector_db.vector_store import VectorStore
from embeddings.embeddings import EmbeddingGenerator


def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_result(result_num, doc_id, similarity, text, metadata):
    """Print formatted search result."""
    print(f"[{result_num}] Similarity: {similarity:.3f} | ID: {doc_id}")
    print(f"    File: {metadata.get('filename', 'N/A')} (Page {metadata.get('page_number', 'N/A')})")
    print(f"    Text: {text[:150]}...")
    if "material_name" in metadata and metadata["material_name"]:
        print(f"    Material: {metadata['material_name']}")
    if "co2_uptake_mmol_g" in metadata and metadata["co2_uptake_mmol_g"]:
        print(f"    CO2 Uptake: {metadata['co2_uptake_mmol_g']} mmol/g")
    print()


def semantic_search():
    """Semantic search."""
    print_header("SEMANTIC SEARCH")
    
    store = VectorStore()
    embedding_gen = EmbeddingGenerator()
    
    while True:
        query = input("Enter query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
        
        if not query:
            print("Empty query. Try again.")
            continue
        
        try:
            top_k = int(input("Number of results (default 5): ") or "5")
            
            print(f"\nSearching for: '{query}'...")
            query_vec = embedding_gen.embed([query])[0].tolist()
            results = store.search(query_vec, top_k=top_k)
            
            print(f"\nFound {len(results['ids'])} results:\n")
            for idx, (doc_id, distance, text, metadata) in enumerate(
                zip(results["ids"], results["distances"], 
                    results["documents"], results["metadatas"]), 1
            ):
                similarity = 1 - distance
                print_result(idx, doc_id, similarity, text, metadata)
        
        except Exception as e:
            print(f"Error: {e}\n")


def browse_documents():
    """Browse documents."""
    print_header("BROWSE DOCUMENTS")
    
    store = VectorStore()
    
    while True:
        limit = int(input("Number of documents to show (default 10): ") or "10")
        
        print(f"\nFetching {limit} documents...")
        results = store.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        for idx, (text, metadata) in enumerate(
            zip(results["documents"], results["metadatas"]), 1
        ):
            print(f"\n[{idx}] {metadata.get('filename', 'N/A')} (Page {metadata.get('page_number', 'N/A')})")
            print(f"    {text[:200]}...")
        
        cont = input("\nShow more? (y/n): ").lower()
        if cont != 'y':
            break


def filter_by_file():
    """Filter documents by filename."""
    print_header("FILTER BY FILE")
    
    store = VectorStore()
    
    # Get unique files
    all_results = store.collection.get(limit=10000, include=["metadatas"])
    files = set(m.get("filename", "unknown") for m in all_results["metadatas"])
    
    print(f"Available files ({len(files)}):")
    file_list = sorted(list(files))
    for idx, f in enumerate(file_list, 1):
        print(f"  {idx}. {f}")
    
    try:
        file_idx = int(input("\nSelect file number: ")) - 1
        if 0 <= file_idx < len(file_list):
            filename = file_list[file_idx]
            
            results = store.collection.where(
                {"filename": {"$eq": filename}}
            )
            
            print(f"\n{len(results['ids'])} chunks from '{filename}':")
            
            for idx, (doc_id, text, metadata) in enumerate(
                zip(results["ids"], results["documents"], results["metadatas"]), 1
            ):
                print(f"\n[{idx}] Page {metadata.get('page_number', 'N/A')}")
                print(f"    {text[:200]}...")
        else:
            print("Invalid selection")
    
    except (ValueError, IndexError):
        print("Invalid input")


def database_stats():
    """Show database statistics."""
    print_header("DATABASE STATISTICS")
    
    store = VectorStore()
    stats = store.get_stats()
    
    print(f"Total Documents: {stats.get('total_documents', 'N/A')}")
    if 'collections' in stats:
        print(f"Collections: {stats['collections']}")
    
    # Get metadata coverage
    all_results = store.collection.get(limit=10000, include=["metadatas"])
    metadatas = all_results["metadatas"]
    
    chemistry_fields = [
        "material_name", "material_class", "co2_uptake_mmol_g",
        "temperature_K", "pressure_bar", "surface_area_m2_g",
        "measurement_method", "source_type", "selectivity"
    ]
    
    print(f"\nChemistry Metadata Coverage:")
    for field in chemistry_fields:
        count = sum(1 for m in metadatas if field in m and m[field])
        percentage = 100 * count / len(metadatas)
        print(f"  {field}: {count} ({percentage:.1f}%)")


def main():
    """Main menu."""
    print_header("VECTOR DATABASE QUERY TOOL")
    
    options = {
        "1": ("Semantic Search", semantic_search),
        "2": ("Browse Documents", browse_documents),
        "3": ("Filter by File", filter_by_file),
        "4": ("Database Statistics", database_stats),
        "5": ("Exit", None),
    }
    
    while True:
        print("\nOptions:")
        for key, (name, _) in options.items():
            print(f"  {key}. {name}")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "5":
            print("\nGoodbye!")
            break
        
        if choice in options:
            _, func = options[choice]
            if func:
                try:
                    func()
                except KeyboardInterrupt:
                    print("\n\nCancelled.")
                except Exception as e:
                    print(f"\nError: {e}")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
