# RAG Pipeline Integration - CO2M Dashboard

Complete Retrieval-Augmented Generation (RAG) pipeline integrated with the CO2M Literature Intelligence Dashboard.

## Overview

This implementation connects a semantic search engine (based on embeddings and vector database) with the Streamlit dashboard, enabling natural language queries across the CO2M corpus.

### Key Components

```
rag_pipeline/
├── __init__.py           # Package exports
├── retriever.py          # RAGRetriever - semantic search engine
└── query_service.py      # RAGQueryService - business logic layer

dashboard/
├── app.py                # Enhanced Streamlit app with RAG tab
├── rag_search.py         # RAG search interface components
└── rag_sidebar.py        # RAG sidebar widget

embeddings/
└── embeddings.py         # EmbeddingGenerator (sentence-transformers)

vector_db/
└── vector_store.py       # VectorStore (Chroma backend)
```

## Quick Start

### 1. Ensure Dependencies

```bash
pip install -r requirements-vector-db.txt
```

### 2. Populate Vector Database

```bash
python ingest_enhanced.py
```

This will:
- Process all PDFs in `co2m/pdfs/`
- Generate embeddings using sentence-transformers
- Index documents in Chroma vector store
- Create `vector_db/data/` with persistent storage

### 3. Verify Setup

```bash
python rag_dashboard_quickstart.py
```

This checks:
- ✓ Dependencies installed
- ✓ Data artifacts available
- ✓ Vector database populated
- ✓ RAG pipeline functional

### 4. Start Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to the **🔍 RAG Search** tab (first tab).

## Using RAG Search

### Basic Query

1. Enter a natural language question
2. Adjust number of results (1-20)
3. Optionally select a concept filter
4. Click "🔍 Search"

### Example Queries

- "What materials are used for CO2 capture?"
- "How do amine-based sorbents work?"
- "What is the efficiency of MOF materials?"
- "Compare zeolites and metal-organic frameworks"
- "Latest advances in CO2 separation"

### Result Interpretation

Each result shows:
- **Rank**: Position in relevance order
- **Similarity**: 0-100% (how relevant)
- **Filename**: Source document
- **Page Number**: Location in PDF
- **Text**: Relevant excerpt (first 300 chars)
- **Metadata**: Material name, CO₂ uptake data if available

### Summary Statistics

- **Results**: Total documents found
- **Unique Docs**: How many source files
- **Avg Similarity**: Average relevance score
- **Materials Found**: Extraction count
- **CO₂ Data**: Min/Max/Avg uptake values

## Python API

### Direct Usage

```python
from rag_pipeline import RAGQueryService

# Initialize
service = RAGQueryService(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_path="./vector_db/data",
    collection_name="co2m_corpus",
    device="cpu",  # or "cuda" for GPU
)

# Execute query
results = service.query(
    question="What materials capture CO2?",
    top_k=5,
    concept_filter=None,  # Optional: "Materials Science"
    include_context=True,
)

# Access results
for result in results['results']:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"File: {result['filename']}")
    print(f"Text: {result['text']}")

# Use formatted context for LLM
llm_context = results['context']
llm_summary = results['context_summary']
```

### Batch Queries

```python
queries = ["CO2 capture", "sorbent materials", "efficiency"]
retriever = service.retriever
batch_results = retriever.batch_retrieve(queries, top_k=5)
```

### Topic Search

```python
results = service.search_by_topic("amine sorbents", top_k=10)
```

### Find Related Documents

```python
sample_text = "Metal-organic frameworks show promise..."
results = service.get_related_documents(sample_text, top_k=5)
```

## Architecture

### Data Flow

```
User Query
    ↓
[Dashboard/CLI Input]
    ↓
RAGQueryService.query()
    ├→ RAGRetriever.retrieve()
    │  ├→ EmbeddingGenerator.embed_single(query)
    │  └→ VectorStore.search(query_embedding)
    │     └→ Chroma.query() [cosine similarity]
    │
    └→ Format Results + Generate Context
    
    ↓
[Results + Context + Summary]
    ↓
[Dashboard Display / LLM Input / Export]
```

### Component Details

#### RAGRetriever
- **Purpose**: Semantic search and document retrieval
- **Input**: Natural language query
- **Output**: Ranked list of relevant documents with similarity scores
- **Methods**:
  - `retrieve(query, top_k, concept_filter)` - Single query
  - `batch_retrieve(queries, top_k, concept_filter)` - Multiple queries
  - `get_stats()` - Vector DB statistics

#### RAGQueryService  
- **Purpose**: Business logic and context generation
- **Input**: Questions, topics, document text
- **Output**: Formatted results with context for LLMs
- **Methods**:
  - `query(question, top_k, concept_filter, include_context)` - Main query
  - `search_by_topic(topic, top_k)` - Topic search
  - `get_related_documents(text, top_k)` - Similarity search
  - `_format_context()` - LLM context formatting
  - `_generate_summary()` - Statistics generation

#### EmbeddingGenerator
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimension**: 384
- **Performance**: Fast, good quality, GPU optional
- **Batch Processing**: Configurable batch size

#### VectorStore
- **Backend**: Chroma (persistent DuckDB)
- **Collection**: co2m_corpus
- **Metric**: Cosine similarity
- **Location**: `./vector_db/data`

## Performance

### Timing (CPU)

| Operation | Time |
|-----------|------|
| First query (model load) | 2-3 sec |
| Subsequent queries | 0.5-1 sec |
| Batch 10 queries | ~5 sec |

### Scaling

- 100 documents: <100ms
- 1,000 documents: 100-200ms
- 10,000+ documents: 200-500ms

### Optimization

- GPU acceleration available (`device='cuda'`)
- Batch processing for multiple queries
- Caching on service initialization
- Efficient embedding model (384-dim, 22M params)

## Troubleshooting

### Vector Database Empty

```
Error: Vector database shows 0 documents
Solution: Run python ingest_enhanced.py
```

### Embedding Model Download

```
Error: Model download fails
Solution: Check internet connection
          Models are from HuggingFace Hub (~300MB)
```

### GPU/CUDA Issues

```
Error: CUDA out of memory or not available
Solution: Set device='cpu' in RAGQueryService()
          or install CPU version of sentence-transformers
```

### Dashboard RAG Tab Missing

```
Error: RAG Search tab not appearing
Solution: Check rag_pipeline/ folder exists
          Verify imports in dashboard/app.py
          Restart streamlit with: streamlit run dashboard/app.py
```

### Irrelevant Results

```
Issue: Search results don't match query well
Solutions: 
  - Try rephrasing query
  - Increase top_k value
  - Verify vector DB is properly populated
  - Check that PDFs were processed (ingest_enhanced.py)
```

## Testing

### Run Test Suite

```bash
python -m pytest tests/test_rag_pipeline.py -v
```

Tests cover:
- Retriever initialization
- Single and batch queries
- Similarity score validation
- Context generation
- Service statistics

### Example Pipeline

```bash
python examples/example_rag_pipeline.py
```

Demonstrates:
- Service initialization
- Multiple example queries
- Result interpretation
- Context and summary usage

## Configuration

### Embedding Settings

```python
RAGQueryService(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Model ID
    device="cpu",  # "cpu" or "cuda"
    batch_size=32,  # Embedding batch size
)
```

### Vector Store Settings

```python
RAGQueryService(
    vector_store_path="./vector_db/data",  # Chroma persistence
    collection_name="co2m_corpus",  # Collection name
)
```

### Query Settings

```python
service.query(
    question="...",
    top_k=5,  # Results to return (1-20)
    concept_filter=None,  # Optional concept
    include_context=True,  # Format for LLM
)
```

## Output Formats

### Query Results

```python
{
    "question": "...",
    "concept_filter": None,
    "num_results": 5,
    "results": [
        {
            "rank": 1,
            "similarity": 0.8234,
            "filename": "paper_001.pdf",
            "page_number": 12,
            "text": "Excerpt from document...",
            "material_name": "MOF-5",
            "co2_uptake_mmol_g": 3.5,
            "metadata": {...}
        },
        # ... more results
    ],
    "context": "[Source 1: ...]\n[Source 2: ...]\n...",
    "context_summary": {
        "total_results": 5,
        "unique_documents": 5,
        "avg_similarity": 0.75,
        "materials_found": 3,
        "co2_uptake_stats": {
            "min": 2.1,
            "max": 4.5,
            "avg": 3.2
        }
    }
}
```

## Integration Points

### With Dashboard

The RAG tab in `dashboard/app.py` provides:
- Natural language search interface
- Real-time result display
- Summary statistics
- Context export for LLMs
- Concept filtering

### With LLM Systems

The formatted context can be used with:
- OpenAI GPT models
- Anthropic Claude
- Open-source models (Llama, etc.)
- Custom language models

Example:

```python
results = service.query("...", include_context=True)
context = results['context']

# Use with LLM
prompt = f"""Based on this context:

{context}

Answer the question: {results['question']}"""

# Send to LLM API
response = llm.query(prompt)
```

## File Structure

```
project_root/
├── rag_pipeline/              # RAG pipeline package
│   ├── __init__.py
│   ├── retriever.py
│   └── query_service.py
│
├── dashboard/
│   ├── app.py                # Main app with RAG tab
│   ├── rag_search.py
│   └── rag_sidebar.py
│
├── vector_db/
│   ├── __init__.py
│   ├── vector_store.py
│   └── data/                 # Chroma persistence
│       └── chroma.sqlite3
│
├── embeddings/
│   ├── __init__.py
│   └── embeddings.py
│
├── examples/
│   └── example_rag_pipeline.py
│
├── tests/
│   └── test_rag_pipeline.py
│
├── co2m/
│   ├── pdfs/                 # Source documents
│   ├── artifacts/            # Processed data
│   └── metadata/
│
├── rag_dashboard_quickstart.py
├── RAG_PIPELINE_GUIDE.md
├── RAG_INTEGRATION_README.md  # This file
└── requirements-vector-db.txt
```

## Dependencies

Key packages:
- `chromadb` - Vector database
- `sentence-transformers` - Embedding model
- `streamlit` - Dashboard framework
- `pandas` - Data handling
- `numpy` - Numerical computing
- `scikit-learn` - ML utilities
- `plotly` - Visualizations

Install all:
```bash
pip install -r requirements-vector-db.txt
```

## License & Attribution

- Vector DB: Chroma (open-source)
- Embeddings: sentence-transformers (HuggingFace)
- Dashboard: Streamlit (open-source)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run `python rag_dashboard_quickstart.py` to verify setup
3. Review `RAG_PIPELINE_GUIDE.md` for detailed documentation
4. Check example in `examples/example_rag_pipeline.py`

---

**Last Updated**: 2024
**Status**: ✅ Production Ready
