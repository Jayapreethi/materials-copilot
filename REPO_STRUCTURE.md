# CO2M Vector Database Repository Structure

## Overview

Cleaned, production-ready repository with only essential scripts for building and querying a chemistry-aware vector database.

## Core Scripts (NECESSARY)

### 1. **embeddings/embeddings.py**
- **Purpose**: Generate text embeddings using sentence-transformers
- **Exports**: `EmbeddingGenerator` class
- **Usage**:
  ```python
  from embeddings.embeddings import EmbeddingGenerator
  gen = EmbeddingGenerator()
  embeddings = gen.embed(["text1", "text2"])  # Returns 384-dim vectors
  ```
- **Deployment**: Required for all embedding operations

### 2. **vector_db/vector_store.py**
- **Purpose**: Chroma vector database interface for storage and retrieval
- **Exports**: `VectorStore` class
- **Key Methods**:
  - `add_documents()` - Store texts with metadata
  - `search()` - Semantic search
  - `delete_by_filename()` - Remove documents
  - `get_stats()` - Database statistics
- **Usage**:
  ```python
  from vector_db.vector_store import VectorStore
  store = VectorStore()
  store.add_documents(texts, embeddings, metadatas, ids)
  results = store.search(query_embedding, top_k=5)
  ```
- **Deployment**: Core storage layer

### 3. **metadata_enricher.py**
- **Purpose**: Extract comprehensive chemistry metadata from text
- **Extracts**:
  - Material identity (name, class, formula, metal center, functional groups, dopants)
  - Structural properties (surface area, pore volume, pore size, density, crystal structure)
  - Performance parameters (CO2 uptake, selectivity, thermal stability, adsorption capacity)
  - Experimental conditions (temperature, pressure, humidity, pH, flow rate, sample mass)
  - Method details (measurement/simulation method, instrument, uncertainty, replicate count)
- **Usage**:
  ```python
  from metadata_enricher import MetadataEnricher
  enricher = MetadataEnricher()
  metadata = enricher.enrich(text, doc_id, chunk_id, filename, page_number)
  ```
- **Deployment**: Required for metadata extraction during ingestion

### 4. **ingestion/ingestion.py**
- **Purpose**: Basic ingestion pipeline (parquet → embeddings → vector store)
- **Note**: Use `ingest_enhanced.py` instead for enriched metadata
- **Deployment**: Fallback option if enhanced ingestion not needed

### 5. **ingest_enhanced.py** ⭐ **MAIN SCRIPT**
- **Purpose**: Production ingestion pipeline with comprehensive chemistry metadata
- **What it does**:
  1. Loads artifact parquet files (chunks, documents, concepts)
  2. Processes each chunk with `MetadataEnricher`
  3. Generates embeddings (384-dimensional)
  4. Stores in Chroma with enriched metadata
  5. Creates manifest with ingestion statistics
- **Run**:
  ```bash
  python ingest_enhanced.py
  ```
- **Output**:
  - Vector database in `vector_db/data/`
  - Manifest in `co2m/artifacts/enhanced_manifest.json`
- **Deployment**: Critical - use this to build the database

### 6. **dashboard.py**
- **Purpose**: Flask web interface for database exploration
- **Features**:
  - Semantic search with results display
  - Browse files and chunks
  - Filter by concepts
  - Statistics dashboard
- **Run**:
  ```bash
  python dashboard.py
  # Visit http://localhost:5000
  ```
- **Deployment**: Optional but recommended for exploration

---

## Configuration Files

### **configs/vector_db.yaml**
- Database configuration (model name, device, batch size)
- Used by ingestion pipeline

### **requirements-vector-db.txt**
- Python dependencies:
  - chromadb
  - sentence-transformers
  - pandas
  - pyyaml
  - flask (for dashboard)

---

## Data Directories

### **co2m/artifacts/** (INPUT)
- `chunks.parquet` - Text chunks with metadata
- `documents.parquet` - Document information
- `concepts.parquet` - CO2M concepts
- `enhanced_manifest.json` - Output manifest (created after ingestion)

### **vector_db/data/** (OUTPUT)
- Chroma database (auto-created)
- Persistent storage for embeddings and metadata

### **co2m/pdfs/** (REFERENCE)
- Original PDF files (not used in processing, reference only)

---

## Metadata Structure

Each vector in database contains:

```json
{
  "id": "paper_018_chunk_04",
  "text": "Full text chunk from PDF...",
  "embedding": [0.12, -0.04, 0.87, ...],  // 384-dim
  "metadata": {
    // Document
    "document_id": "paper_018",
    "filename": "core_8366015.pdf",
    "section": "Results",
    "page": 8,
    
    // Material Identity
    "material_name": "UiO-66-NH2",
    "material_class": "MOF",
    "chemical_formula": "Zr6O4(OH)4(BDC-NH2)6",
    "metal_center": "Zr",
    "functional_groups": "NH2",
    "dopants": null,
    
    // Structural Properties
    "surface_area_m2_g": 1120,
    "pore_volume_cm3_g": 0.48,
    "pore_size_nm": 0.72,
    "density_g_cm3": 0.89,
    "crystal_structure": "cubic",
    
    // Performance Parameters
    "co2_uptake_mmol_g": 4.2,
    "selectivity": 15.5,
    "heat_of_adsorption_kJ_mol": 35,
    "conversion_percent": null,
    "stability_cycles": 50,
    
    // Experimental Conditions
    "temperature_K": 298,
    "pressure_bar": 1.0,
    "humidity_percent": 0,
    "ph": null,
    "concentration": null,
    "flow_rate": null,
    "sample_mass": null,
    
    // Method & Evidence
    "measurement_method": "volumetric adsorption",
    "simulation_method": null,
    "instrument": "BELSOPR",
    "uncertainty_percent": 5,
    "replicate_count": 3,
    "table_or_figure_reference": "Table 1",
    
    // Source & Classification
    "source_type": "experimental",
    "gas": "CO2",
    "gas_composition": "pure CO2",
    
    // CO2M Integration
    "concepts": "Carbon capture,MOF",
    "max_confidence": 0.85
  }
}
```

---

## Quick Start Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements-vector-db.txt
```

### Step 2: Build Vector Database
```bash
python ingest_enhanced.py
```
- Processes ~3,689 chunks
- Takes 2-3 minutes
- Expected coverage: 66% materials, 51% measurements, 45% conditions

### Step 3: Explore Database

**Option A: Web Interface**
```bash
python dashboard.py
# Open http://localhost:5000
```

**Option B: Python API**
```python
from vector_db.vector_store import VectorStore
from embeddings.embeddings import EmbeddingGenerator

store = VectorStore()
gen = EmbeddingGenerator()

# Get stats
print(store.get_stats())

# Semantic search
query_vec = gen.embed(["CO2 capture with high uptake"])[0].tolist()
results = store.search(query_vec, top_k=10)

for doc_id, distance, text, metadata in zip(
    results["ids"],
    results["distances"],
    results["documents"],
    results["metadatas"],
):
    similarity = 1 - distance
    print(f"[{similarity:.2%}] {metadata.get('material_name')} - {metadata.get('co2_uptake_mmol_g')} mmol/g")
    print(f"  Conditions: T={metadata.get('temperature_K')}K, P={metadata.get('pressure_bar')}bar")
    print(f"  Method: {metadata.get('measurement_method')}")
    print(f"  Text: {text[:100]}...\n")
```

---

## Common Tasks

### Task: Search for specific materials
```python
from vector_db.vector_store import VectorStore

store = VectorStore()
results = store.collection.get(limit=10000, include=["metadatas"])
mofs = [m for m in results["metadatas"] if m.get("material_class") == "MOF"]
print(f"Found {len(mofs)} MOF entries")
```

### Task: Export high-performance materials
```python
results = store.collection.get(limit=10000, include=["metadatas"])
high_uptake = [
    m for m in results["metadatas"] 
    if m.get("co2_uptake_mmol_g", 0) > 5.0
]
```

### Task: Find computational vs experimental
```python
results = store.collection.get(limit=10000, include=["metadatas"])
experimental = [m for m in results["metadatas"] if m.get("source_type") == "experimental"]
computational = [m for m in results["metadatas"] if m.get("source_type") == "computational"]
print(f"Experimental: {len(experimental)}, Computational: {len(computational)}")
```

### Task: Analyze extraction coverage
```python
from ingest_enhanced import EnhancedIngestionPipeline

pipeline = EnhancedIngestionPipeline()
results = pipeline.vector_store.collection.get(limit=10000, include=["metadatas"])
pipeline.validate_metadata(results["metadatas"])
```

---

## Metadata Coverage Statistics

Expected coverage for 3,689 chunks:

| Field | Coverage | Notes |
|-------|----------|-------|
| Material Name | ~67% | Auto-extracted using patterns |
| Material Class | ~67% | MOF, Zeolite, Carbon, etc. |
| Chemical Formula | ~40% | Regex-based extraction |
| Surface Area | ~23% | m²/g from BET |
| Pore Volume | ~15% | cm³/g from BJH |
| Pore Size | ~12% | Nanometer scale |
| CO2 Uptake | ~51% | mmol/g quantitative |
| Temperature | ~45% | Kelvin, most conditions |
| Pressure | ~40% | Bar, adsorption studies |
| Method | ~55% | Volumetric, gravimetric, etc. |
| Source Type | ~70% | Experimental vs computational |

---

## Troubleshooting

### Issue: "No vectors added to collection"
**Solution**: 
- Check parquet files exist: `ls co2m/artifacts/*.parquet`
- Run `ingest_enhanced.py` with verbose logging
- Verify metadata_enricher.py is not returning all None values

### Issue: Low metadata coverage
**Solution**:
- This is expected! Coverage ~50% is normal for scientific text
- Fields only populate if patterns match the text
- Patterns are conservative to avoid false positives

### Issue: Dashboard not responding
**Solution**:
- Check port 5000 is free: `netstat -ano | findstr :5000`
- Try different port: `python dashboard.py --port 5001`
- Check Flask is installed: `pip install flask`

### Issue: Out of memory during ingestion
**Solution**:
- Reduce batch size in `ingest_enhanced.py`
- Process chunks in smaller groups
- Use SSD instead of HDD

---

## Production Deployment

### Step 1: Build Database Once
```bash
python ingest_enhanced.py
# Creates vector_db/data/ (persistent)
```

### Step 2: Run Dashboard in Background
```bash
# Linux/Mac
nohup python dashboard.py > dashboard.log 2>&1 &

# Windows
start python dashboard.py

# Docker (recommended)
docker run -p 5000:5000 -v $(pwd)/vector_db:/app/vector_db my-co2m-dashboard
```

### Step 3: Query via API
```bash
# Health check
curl http://localhost:5000/api/stats

# Search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CO2 capture MOF", "top_k": 10}'
```

---

## File Manifest

```
.
├── embeddings/
│   ├── __init__.py
│   └── embeddings.py              ⭐ Core embedding generation
├── vector_db/
│   ├── __init__.py
│   ├── vector_store.py            ⭐ Core database interface
│   └── data/                       (created after ingestion)
├── ingestion/
│   ├── __init__.py
│   └── ingestion.py               (optional basic ingestion)
├── configs/
│   └── vector_db.yaml             Configuration
├── co2m/
│   ├── artifacts/
│   │   ├── chunks.parquet         INPUT
│   │   ├── documents.parquet      INPUT
│   │   ├── concepts.parquet       INPUT
│   │   └── enhanced_manifest.json OUTPUT
│   ├── logs/
│   ├── metadata/
│   └── pdfs/                      (reference)
├── dashboard.py                   ⭐ Web interface (optional)
├── ingest_enhanced.py             ⭐ Main ingestion script
├── metadata_enricher.py           ⭐ Metadata extraction engine
├── ENHANCED_FORMAT.md             Documentation
├── ENHANCED_QUICKSTART.md         Quick start guide
├── requirements-vector-db.txt     Dependencies
└── REPO_STRUCTURE.md              This file
```

---

## Summary

**Necessary Scripts (MUST HAVE):**
1. ✅ `metadata_enricher.py` - Chemistry metadata extraction
2. ✅ `ingest_enhanced.py` - Build database with enriched metadata
3. ✅ `embeddings/embeddings.py` - Text embeddings
4. ✅ `vector_db/vector_store.py` - Vector database interface

**Recommended Scripts (SHOULD HAVE):**
5. ✅ `dashboard.py` - Web exploration interface

**Optional Scripts:**
- `ingestion/ingestion.py` - Basic ingestion without enrichment
- Custom Python scripts using the core modules

**Removed (Redundant):**
- ❌ `cli_viewer.py` - CLI viewer (dashboard is better)
- ❌ `verify_vector_db.py` - Verification (integrated into pipeline)
- ❌ `query_enhanced.py` - Query tool (use API/Python directly)
- ❌ `vector_db_crud.py` - Documentation only
- ❌ `view_chroma.py` - Old viewer
- ❌ Old example scripts
- ❌ Old dashboard code (`co2m_lit_dashboard/`)
- ❌ Old RAG code (`rag_pipeline/`)
