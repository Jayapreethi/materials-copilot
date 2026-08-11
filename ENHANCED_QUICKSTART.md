# Enhanced Database - Quick Start

## TL;DR

```bash
# 1. Build enhanced database (2-3 min)
python ingest_enhanced.py

# 2. Query it
python query_enhanced.py

# 3. Or use the dashboard
python dashboard.py  # Visit http://localhost:5000
```

## What's New

Your vector database now has **rich structured metadata** automatically extracted from the PDF corpus:

```
Old Format:
- chunk_id, doc_id, text, embedding, filename, page_number

New Format:
- chunk_id, doc_id, text, embedding, PLUS:
  ✓ Material name & class & formula
  ✓ Measurement type & value & unit
  ✓ Temperature, pressure, humidity
  ✓ Surface area, pore volume, pore size
  ✓ Experimental method & gas type
  ✓ Concepts & confidence scores
```

## Step 1: Build Enhanced Database

```bash
python ingest_enhanced.py
```

**What happens:**
1. Loads 3,689 chunks from parquet files
2. Extracts materials, measurements, conditions using pattern matching
3. Generates embeddings (takes ~2 min)
4. Stores in Chroma with rich metadata
5. Saves manifest with extraction stats

**Expected stats:**
```
Metadata Statistics:
  Total chunks: 3,689
  With material info: ~2,456 (66.6%)
  With measurements: ~1,892 (51.3%)
  With conditions: ~1,645 (44.6%)
  With properties: ~856 (23.2%)
```

## Step 2: Query the Database

### Option A: Interactive CLI

```bash
python query_enhanced.py
```

Menu options:
```
[1] Semantic Search         - "What materials capture CO2?"
[2] Filter by Material      - "Show me all MOFs"
[3] Filter by Measurement   - "Find CO2 uptake studies"
[4] Filter by Gas          - "Which studies used N2?"
[5] Filter by Conditions   - "Experiments at 298K"
[6] Materials Summary      - "How many MOF chunks?"
[7] Measurements Summary   - "What's measured most?"
```

Example session:
```
Select option: 1
Enter query: carbon capture with high uptake

Results (5):
[1] paper_018 (Page 8, Similarity: 0.587)
    Material: UiO-66-NH2 (MOF)
    Measurement: 4.2 mmol/g CO2 uptake
    Text: "The material achieved a CO2 uptake..."

[2] paper_042 (Page 3, Similarity: 0.571)
    Material: MIL-101 (MOF)
    Measurement: 3.8 mmol/g CO2 uptake
    Text: "Excellent CO2 capture performance..."
```

### Option B: Python API

```python
from query_enhanced import EnhancedQuery

q = EnhancedQuery()

# Search
results = q.semantic_search("MOF carbon capture", top_k=3)
for r in results:
    print(f"{r['document_id']}: {r['material']['name']} - {r['measurement']['value']} {r['measurement']['unit']}")

# Filter by material class
moofs = q.filter_by_material("MOF", limit=10)
print(f"Found {len(moofs)} MOF results")

# Filter by conditions
room_temp = q.filter_by_conditions(
    temperature_range=(298, 323),  # 25-50°C
    limit=10
)

# Get summaries
materials = q.get_materials_summary()
print(f"Materials: {materials['materials']}")
# Output: {'MOF': 892, 'Zeolite': 345, 'Activated Carbon': 234, ...}
```

### Option C: Web Dashboard

```bash
python dashboard.py
```

Visit: **http://localhost:5000**

Features:
- Search by semantic meaning
- Browse files
- Filter by concepts
- View random samples

## Metadata Fields

Every chunk now has:

**Document Info:**
- `document_id` - e.g., "paper_018"
- `filename` - e.g., "core_8366015.pdf"
- `section` - "abstract", "methods", "results", etc.
- `page` - page number

**Material:**
- `material_name` - e.g., "UiO-66-NH2"
- `material_class` - e.g., "MOF"
- `material_formula` - e.g., "Zr6O4(OH)4(BDC-NH2)6"

**Measurement:**
- `measurement` - "CO2 uptake", "N2 uptake", "Surface area", etc.
- `value` - 4.2, 1120, etc.
- `unit` - "mmol/g", "m2/g", etc.

**Conditions:**
- `temperature_K` - 298, 323, etc.
- `pressure_bar` - 1.0, 10.0, etc.
- `humidity_percent` - 0, 50, etc.

**Properties:**
- `surface_area_m2_g` - 1120, 2500, etc.
- `pore_volume_cm3_g` - 0.48, 0.85, etc.
- `pore_size_nm` - 0.72, 1.5, etc.

**Method:**
- `method` - "volumetric adsorption", "GCMC", etc.
- `gas` - "CO2", "N2", "CH4", etc.
- `source_type` - "experimental" or "computational"

**CO2M:**
- `concepts` - CO2M concept tags
- `max_confidence` - 0-1 score

## Common Queries

```python
from query_enhanced import EnhancedQuery

q = EnhancedQuery()

# Find all MOF studies with high CO2 uptake
mofs = q.filter_by_material("MOF", limit=50)
high_uptake = [r for r in moofs if r.get('value', 0) > 3.0]

# Find computational studies
all_results = q.store.collection.get(limit=10000, include=["metadatas"])
computational = [m for m in all_results["metadatas"] if m.get("source_type") == "computational"]

# Find room temperature experiments
room_temp = q.filter_by_conditions(temperature_range=(298, 298), limit=50)

# CO2 uptake rankings
co2_results = q.filter_by_measurement("CO2 uptake", limit=100)
ranked = sorted(co2_results, key=lambda x: x.get('measurement', {}).get('value', 0), reverse=True)
for r in ranked[:10]:
    print(f"{r['material']}: {r['measurement']['value']} {r['measurement']['unit']}")
```

## API Endpoints (Dashboard)

```bash
# Get stats
curl http://localhost:5000/api/stats

# Search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CO2 capture", "top_k": 5}'

# Get files
curl http://localhost:5000/api/files

# Get file chunks
curl http://localhost:5000/api/file/core_8366015.pdf

# Filter by concept
curl http://localhost:5000/api/concept/Carbon%20capture
```

## Troubleshooting

**No metadata extracted?**
```
- Check if text contains keywords (see ENHANCED_FORMAT.md)
- Material: "MOF", "zeolite", "carbon", etc.
- Measurement: "uptake", "BET", "volume", etc.
- Condition: "298 K", "1 bar", etc.
```

**Search too slow?**
```bash
# Reduce top_k in query
q.semantic_search(query, top_k=3)  # Instead of 10
```

**Database not building?**
```bash
# Check artifacts exist
ls co2m/artifacts/
# Should have: chunks.parquet, documents.parquet, concepts.parquet
```

## Files Created

| File | Purpose |
|------|---------|
| `metadata_enricher.py` | Auto-extracts structured metadata |
| `ingest_enhanced.py` | Builds enhanced database from artifacts |
| `query_enhanced.py` | Interactive query tool |
| `ENHANCED_FORMAT.md` | Full documentation |
| `VECTOR_DB_GUIDE.md` | General vector DB guide |

## Performance

| Task | Time |
|------|------|
| Build database | 2-3 minutes |
| Semantic search | <500ms |
| Filter by material | <1s |
| Get materials summary | <2s |

## Next Steps

1. **Explore data:** `python query_enhanced.py` → Try all filter options
2. **Analyze results:** See which materials/measurements are most common
3. **Build applications:** Use Python API for custom analysis
4. **Export data:** Extend query tool to export to CSV/JSON

## Example Analysis

```python
from query_enhanced import EnhancedQuery
import json

q = EnhancedQuery()

# Get all data
results = q.store.collection.get(limit=10000, include=["metadatas"])
metadatas = results["metadatas"]

# Analyze materials
materials = {}
for m in metadatas:
    mat = m.get("material_name", "unknown")
    if mat not in materials:
        materials[mat] = {"count": 0, "avg_uptake": 0}
    materials[mat]["count"] += 1
    if m.get("value"):
        materials[mat]["avg_uptake"] = (materials[mat]["avg_uptake"] + m["value"]) / 2

# Top 10 materials
top = sorted(materials.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
print(json.dumps(top, indent=2))

# CO2 vs N2 comparison
co2_count = sum(1 for m in metadatas if m.get("gas") == "CO2")
n2_count = sum(1 for m in metadatas if m.get("gas") == "N2")
print(f"CO2 studies: {co2_count}, N2 studies: {n2_count}")
```

---

**Ready to explore?** Start with `python query_enhanced.py` 🚀
