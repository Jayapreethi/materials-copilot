# Vector Database Quick Reference

## API Overview

### Initialize
```python
from vector_db.vector_store import VectorStore
from embeddings.embeddings import EmbeddingGenerator

store = VectorStore(persist_directory="./vector_db/data")
embedding_gen = EmbeddingGenerator()
```

---

## 1. CREATE - Add Documents

### Basic Add
```python
texts = ["Document text chunk 1", "Document text chunk 2"]
embeddings = embedding_gen.embed(texts)

metadatas = [
    {
        "chunk_id": 1,
        "doc_id": 100,
        "filename": "myfile.pdf",
        "page_number": 1,
        "concept_names": "Climate,Energy",
        "max_confidence": 0.85
    },
    {
        "chunk_id": 2,
        "doc_id": 100,
        "filename": "myfile.pdf",
        "page_number": 2,
        "concept_names": "Climate",
        "max_confidence": 0.92
    }
]

store.add_documents(
    texts=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=["doc_100_1", "doc_100_2"]
)
```

### Metadata Fields (Required)
- `chunk_id` (int) - Unique chunk identifier
- `doc_id` (int) - Document identifier
- `filename` (str) - PDF filename
- `page_number` (int) - Page in document
- `concept_names` (str) - Comma-separated concepts
- `max_confidence` (float) - Confidence score (0-1)

---

## 2. READ - Query Documents

### Semantic Search
```python
query = "carbon sequestration"
query_embedding = embedding_gen.embed([query])[0].tolist()

results = store.search(
    query_embedding=query_embedding,
    top_k=5
)

# Access results
for doc_id, distance, text, metadata in zip(
    results['ids'],
    results['distances'],
    results['documents'],
    results['metadatas']
):
    similarity = 1 - distance  # Convert to similarity score
    print(f"Score: {similarity:.3f}")
    print(f"Text: {text}")
    print(f"File: {metadata['filename']}")
```

### With Metadata Filter
```python
where_filter = {"filename": {"$eq": "specific_file.pdf"}}

results = store.search(
    query_embedding=query_embedding,
    top_k=10,
    where=where_filter
)
```

### Get All from File
```python
results = store.collection.get(
    where={"filename": {"$eq": "myfile.pdf"}},
    include=["documents", "metadatas"]
)
```

### Get Statistics
```python
stats = store.get_stats()
# Returns: collection_name, total_documents, distance_metric, persist_directory
```

---

## 3. UPDATE - Modify Documents

### Method: Delete + Re-add
```python
# Step 1: Delete old
deleted = store.delete_by_filename("old_file.pdf")

# Step 2: Add new with same doc_id but updated content
store.add_documents(
    texts=new_texts,
    embeddings=embeddings.tolist(),
    metadatas=new_metadatas,
    ids=new_ids
)
```

---

## 4. DELETE - Remove Documents

### Delete by Filename
```python
deleted_count = store.delete_by_filename("unwanted_file.pdf")
```

### Delete by IDs
```python
store.collection.delete(ids=["doc_1_chunk_1", "doc_1_chunk_2"])
```

### Delete by Concept
```python
results = store.collection.get(limit=10000)
to_delete = [
    doc_id for doc_id, meta in zip(results["ids"], results["metadatas"])
    if "Climate" in meta["concept_names"]
]
store.collection.delete(ids=to_delete)
```

---

## Important Notes

### Metadata Constraints
- ❌ Cannot use `None` values - omit key instead
- ❌ Cannot use lists - use comma-separated strings
- ❌ Cannot use empty lists
- ✅ Use strings for concepts: `"Climate,Energy,Technology"`

### Similarity Score
- Score ranges: 0.0 to 1.0
- 1.0 = Perfect match
- 0.5 = Medium relevance
- 0.0 = No match
- Convert from distance: `similarity = 1 - distance`

### Batch Operations
- Add multiple documents at once (faster)
- All lists must have same length
- IDs must be unique across database

### Persistence
- Changes saved automatically to `./vector_db/data`
- No manual `persist()` call needed
- Safe to interrupt after add/delete

---

## Common Patterns

### Search + Filter
```python
query_embedding = embedding_gen.embed(["my query"])[0].tolist()
results = store.search(
    query_embedding=query_embedding,
    top_k=10,
    where={"concept_names": {"$contains": "Climate"}}  # NOT SUPPORTED
)
# Alternative: Filter in Python
filtered = [r for r in results if "Climate" in r['concept_names']]
```

### Bulk Add from DataFrame
```python
import pandas as pd

df = pd.read_csv("chunks.csv")
texts = df["text"].tolist()
embeddings = embedding_gen.embed(texts)

metadatas = []
for _, row in df.iterrows():
    metadatas.append({
        "chunk_id": int(row["chunk_id"]),
        "doc_id": int(row["doc_id"]),
        "filename": str(row["filename"]),
        "page_number": int(row["page_number"]),
        "concept_names": str(row["concepts"]),
        "max_confidence": float(row["confidence"])
    })

ids = [f"{m['doc_id']}_{m['chunk_id']}" for m in metadatas]

store.add_documents(
    texts=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)
```

### Retrieve with Concepts
```python
from rag_pipeline.retriever import Retriever

retriever = Retriever(
    embedding_generator=embedding_gen,
    vector_store=store,
    top_k=5
)

# Search with optional concept filter
results = retriever.retrieve(
    query="carbon capture technology",
    concept_filter="Carbon capture"
)
```

---

## Run Examples

```bash
# CRUD operations
python vector_db_crud.py

# CLI viewer
python cli_viewer.py

# Quick verification
python verify_vector_db.py

# View sample
python view_chroma.py
```
