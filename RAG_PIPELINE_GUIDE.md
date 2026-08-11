"""
RAG Pipeline Integration Guide
Complete walkthrough of the RAG pipeline with the dashboard.
"""

# ============================================================================
# QUICK START
# ============================================================================

"""
1. ENSURE VECTOR DATABASE IS POPULATED
   
   python ingest_enhanced.py
   
   This indexes all PDFs into the vector store. You should see:
   - Embeddings generated
   - Documents added to Chroma
   - Manifest updated

2. START THE DASHBOARD

   streamlit run dashboard/app.py
   
   The dashboard will now have a "🔍 RAG Search" tab as the first tab.

3. USE RAG SEARCH

   - Enter natural language queries
   - Adjust number of results (1-20)
   - Optionally filter by concept
   - View similarity scores and formatted context

"""

# ============================================================================
# COMPONENTS
# ============================================================================

"""
1. RAG PIPELINE MODULES (rag_pipeline/)

   ├── __init__.py           - Package initialization
   ├── retriever.py          - RAGRetriever class
   │   └── .retrieve(query, top_k, concept_filter)
   │   └── .batch_retrieve(queries, top_k, concept_filter)
   │   └── .get_stats()
   │
   └── query_service.py      - RAGQueryService class
       ├── .query(question, top_k, concept_filter, include_context)
       ├── .search_by_topic(topic, top_k)
       ├── .get_related_documents(text, top_k)
       └── .get_stats()

2. DASHBOARD INTEGRATION

   ├── dashboard/app.py          - Main Streamlit app with RAG tab
   ├── dashboard/rag_search.py   - RAG search interface
   └── dashboard/rag_sidebar.py  - RAG sidebar widget

3. SUPPORTING COMPONENTS

   ├── embeddings/embeddings.py  - EmbeddingGenerator
   ├── vector_db/vector_store.py - VectorStore (Chroma)
   └── ingestion/ingestion.py    - IngestionPipeline

"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
DIRECT PYTHON USAGE:

    from rag_pipeline import RAGQueryService
    
    # Initialize service
    service = RAGQueryService()
    
    # Execute a query
    results = service.query(
        question="What materials are best for CO2 capture?",
        top_k=5,
        include_context=True
    )
    
    # Access results
    for result in results['results']:
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"File: {result['filename']}")
        print(f"Text: {result['text']}")
    
    # Use formatted context for LLM
    context = results['context']
    summary = results['context_summary']

BATCH QUERIES:

    queries = [
        "CO2 capture mechanisms",
        "Sorbent materials",
        "Efficiency metrics"
    ]
    
    retriever = RAGQueryService().retriever
    results = retriever.batch_retrieve(queries, top_k=5)

CONCEPT FILTERING:

    results = service.query(
        question="What materials...",
        concept_filter="Materials Science"  # Filter by concept
    )

"""

# ============================================================================
# ARCHITECTURE FLOW
# ============================================================================

"""
USER QUERY
    ↓
[Dashboard / CLI]
    ↓
RAGQueryService.query()
    ↓
RAGRetriever.retrieve()
    ↓
EmbeddingGenerator.embed_single(query)
    ↓
VectorStore.search(query_embedding)
    ↓
Chroma.query() → cosine similarity search
    ↓
RESULTS
    ├─ Raw retrieval results
    ├─ Formatted context
    └─ Summary statistics

"""

# ============================================================================
# CONFIGURATION
# ============================================================================

"""
EMBEDDING MODEL:
  sentence-transformers/all-MiniLM-L6-v2
  - Dimension: 384
  - Performance: Fast, good quality
  - GPU Support: Yes

VECTOR STORE:
  - Backend: Chroma (persistent DuckDB)
  - Location: ./vector_db/data
  - Collection: co2m_corpus
  - Metric: Cosine similarity

RETRIEVAL:
  - Top-K: Configurable per query (1-20)
  - Similarity Range: [0, 1]
  - Filtering: By concept (optional)

"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Q: Vector database shows 0 documents
A: Run: python ingest_enhanced.py

Q: Embedding model download fails
A: Check internet connection. Models are downloaded from HuggingFace Hub.
   
Q: GPU/CUDA issues
A: Set device='cpu' in RAGQueryService initialization.
   
Q: Dashboard RAG tab not appearing
A: Ensure rag_pipeline/ folder is in Python path.
   Check imports in dashboard/app.py

Q: Search results are irrelevant
A: Try different query phrasing or adjust top_k value.
   Verify vector database is properly populated.

"""

# ============================================================================
# PERFORMANCE NOTES
# ============================================================================

"""
EMBEDDING:
  - First query: ~2-3 seconds (model load + embedding)
  - Subsequent queries: ~0.5-1 second
  - Batch: Linear with query count

RETRIEVAL:
  - 100 documents: <100ms
  - 1000 documents: ~100-200ms
  - 10000+ documents: ~200-500ms

OPTIMIZATION:
  - Device='cpu' is fine for most queries
  - GPU acceleration available with device='cuda'
  - Batch processing for multiple queries
  - LRU cache on service initialization

"""
