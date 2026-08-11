# Enhanced Vector Database Format

## Overview

The vector database now uses a **structured metadata format** that automatically extracts and enriches data from CO2M corpus documents. Each entry contains not just text and embeddings, but rich structured information about materials, measurements, experimental conditions, and properties.

## Format Structure

```json
{
  "id": "paper_018_chunk_04",
  "text": "The material achieved a CO2 uptake of 4.2 mmol/g at 298 K and 1 bar...",
  "embedding": [0.12, -0.04, 0.87, ...],  // 384-dimensional
  "metadata": {
    // Document Info
    "document_id": "paper_018",
    "filename": "core_8366015.pdf",
    "section": "Results",
    "page": 8,
    
    // Material Information
    "material_name": "UiO-66-NH2",
    "material_class": "MOF",
    "material_formula": "Zr6O4(OH)4(BDC-NH2)6",
    
    // Measurement Data
    "measurement": "CO2 uptake",
    "value": 4.2,
    "unit": "mmol/g",
    
    // Experimental Conditions
    "temperature_K": 298,
    "pressure_bar": 1.0,
    "humidity_percent": 0,
    
    // Material Properties
    "surface_area_m2_g": 1120,
    "pore_volume_cm3_g": 0.48,
    "pore_size_nm": 0.72,
    
    // Method & Source
    "method": "volumetric adsorption",
    "gas": "CO2",
    "source_type": "experimental",
    
    // CO2M Concepts
    "concepts": "Carbon capture,MOF",
    "max_confidence": 0.85
  }
}
```

## Auto-Extracted Metadata Fields

### 1. Document Info
- `document_id` - Unique paper ID (auto-generated from doc_id)
- `filename` - Source PDF filename
- `section` - Document section (abstract, methods, results, etc.)
- `page` - Page number in document

### 2. Material Information
- `material_name` - Specific material name (e.g., UiO-66-NH2)
- `material_class` - Material category:
  - MOF (Metal-Organic Framework)
  - Zeolite
  - Activated Carbon
  - Polymer
  - Silica
  - Graphene
  - Aerogel
  - Resin
  - Composite
- `material_formula` - Chemical formula (e.g., Zr6O4(OH)4(BDC-NH2)6)

### 3. Measurement Data
- `measurement` - Type of measurement:
  - CO2 uptake
  - N2 uptake
  - Surface area
  - Pore volume
  - Pore size
  - Thermal stability
- `value` - Numerical measurement value
- `unit` - Unit of measurement (mmol/g, m2/g, cm3/g, nm, °C, etc.)

### 4. Experimental Conditions
- `temperature_K` - Temperature in Kelvin
- `pressure_bar` - Pressure in bar
- `humidity_percent` - Relative humidity percentage
- `pH` - pH value (if applicable)

### 5. Material Properties
- `surface_area_m2_g` - Surface area in m²/g (BET method)
- `pore_volume_cm3_g` - Pore volume in cm³/g (BJH method)
- `pore_size_nm` - Pore diameter in nanometers

### 6. Method & Source
- `method` - Experimental method:
  - volumetric adsorption
  - gravimetric adsorption
  - breakthrough
  - isotherm
  - TGA
  - etc.
- `gas` - Type of gas (CO2, N2, CH4, O2, H2, etc.)
- `source_type` - "experimental" or "computational"

### 7. CO2M Taxonomy
- `concepts` - CO2M concept tags (comma-separated)
- `max_confidence` - Confidence score (0-1)

## Building Enhanced Database

### Step 1: Run Enhanced Ingestion

```bash
python ingest_enhanced.py
```

This will:
1. Load parquet artifacts from `co2m/artifacts/`
2. Process 3,689 chunks with metadata enrichment
3. Generate embeddings (384-dimensional)
4. Store in Chroma with structured metadata
5. Save enhanced manifest: `co2m/artifacts/enhanced_manifest.json`

Expected output:
```
[INFO] Starting enhanced ingestion of 3689 chunks...
[INFO] Processing chunk 500/3689...
[INFO] Generating embeddings for batch...
[INFO] Metadata Statistics:
  Total chunks: 3689
  With material info: 2456 (66.6%)
  With measurements: 1892 (51.3%)
  With conditions: 1645 (44.6%)
  With properties: 856 (23.2%)
```

### Step 2: Verify Enhancement

```bash
python query_enhanced.py
```

Select option [6] or [7] to see materials and measurements summary.

## Querying Enhanced Database

### Option 1: Interactive Query Tool

```bash
python query_enhanced.py
```

Features:
- [1] Semantic search with full metadata display
- [2] Filter by material class
- [3] Filter by measurement type
- [4] Filter by gas type
- [5] Filter by experimental conditions
- [6] View materials summary
- [7] View measurements summary

### Option 2: Python API

```python
from query_enhanced import EnhancedQuery

query = EnhancedQuery()

# Semantic search
results = query.semantic_search("CO2 capture using MOFs", top_k=5)
for r in results:
    print(f"Material: {r['material']['name']}")
    print(f"Uptake: {r['measurement']['value']} {r['measurement']['unit']}")
    print(f"Temp: {r['conditions']['temperature_K']}K")

# Filter by material
mof_results = query.filter_by_material("MOF", limit=10)

# Filter by measurement
uptake_results = query.filter_by_measurement("CO2 uptake", limit=10)

# Filter by conditions
results = query.filter_by_conditions(
    temperature_range=(298, 323),  # 25-50°C
    pressure_range=(0.5, 2.0),     # 0.5-2 bar
    limit=10
)

# Get summaries
materials = query.get_materials_summary()
measurements = query.get_measurements_summary()
```

### Option 3: REST API

The dashboard also works with enhanced format:

```bash
# Start dashboard
python dashboard.py

# Search
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "carbon capture MOF", "top_k": 5}'
```

All metadata fields are accessible in results.

## Extraction Examples

### Example 1: CO2 Uptake Measurement

**Text:**
```
The material UiO-66-NH2 achieved a CO2 uptake of 4.2 mmol/g 
at 298 K and 1 bar with a BET surface area of 1120 m2/g.
```

**Extracted Metadata:**
```
material_name: "UiO-66-NH2"
material_class: "MOF"
measurement: "CO2 uptake"
value: 4.2
unit: "mmol/g"
temperature_K: 298
pressure_bar: 1.0
surface_area_m2_g: 1120
method: "volumetric adsorption"
gas: "CO2"
```

### Example 2: Computational Study

**Text:**
```
GCMC simulations of N2 adsorption on activated carbon show 
excellent agreement with experimental data. Pore volume: 0.5 cm³/g
```

**Extracted Metadata:**
```
measurement: "N2 uptake"
source_type: "computational"
pore_volume_cm3_g: 0.5
method: "GCMC"
gas: "N2"
```

## Metadata Coverage

Based on 3,689 chunks:

| Field | Coverage | Notes |
|-------|----------|-------|
| Material Name | ~67% | Automatically extracted from text |
| Material Class | ~67% | Identified from keywords (MOF, Zeolite, etc.) |
| Measurement | ~51% | Detected from common patterns |
| Temperature | ~45% | Usually reported in experimental sections |
| Pressure | ~40% | Common in adsorption studies |
| Surface Area | ~23% | Often omitted in abstracts |
| Pore Volume | ~15% | More detailed property |
| Pore Size | ~12% | Specialized measurements |

## Performance

- **Ingestion Time**: ~2-3 minutes for 3,689 chunks
- **Query Speed**: <500ms for semantic search
- **Memory**: ~2GB RAM
- **Disk Space**: ~500MB for database

## Enrichment Rules

### Material Class Detection
```
MOF: "metal organic framework", "UiO", "MIL", "HKUST", "ZIF"
Zeolite: "zeolite", "13X", "5A", "mordenite"
Activated Carbon: "activated carbon", "AC", "charcoal"
Polymer: "polymer", "polyamide", "polystyrene"
Silica: "silica", "SiO2", "MCM"
Graphene: "graphene", "GO"
Aerogel: "aerogel", "cryogel"
Resin: "resin", "ion-exchange"
Composite: "composite", "hybrid", "nanocomposite"
```

### Measurement Type Detection
```
CO2 uptake: "CO2 uptake|adsorption|capture|absorption"
N2 uptake: "N2 uptake|adsorption"
Surface area: "surface area|BET"
Pore volume: "pore volume|BJH"
Pore size: "pore size|pore diameter"
Thermal stability: "thermal stability|decomposition"
```

### Method Detection
```
"volumetric adsorption", "gravimetric", "breakthrough",
"isotherm", "TGA", "calorimetry", "spectroscopy", etc.
```

## Future Enhancements

- [ ] ML-based material name recognition
- [ ] Structured table extraction from PDFs
- [ ] Chemical similarity matching
- [ ] Performance prediction models
- [ ] Experimental condition optimization
- [ ] Publication metadata enrichment
- [ ] Citation extraction
- [ ] Author expertise tagging

## Troubleshooting

### Low Coverage for Specific Field

If some fields have low coverage:

1. Check if pattern needs updating in `metadata_enricher.py`
2. Review extraction accuracy with sample texts
3. Consider rule-based extraction for key fields
4. Use fallback values for missing data

### Incorrect Extraction

Example: Temperature extracted from unrelated number

Solution: Improve regex patterns in `CONDITION_PATTERNS`

```python
# Instead of:
r"(\d+\.?\d*)\s*K"

# Use:
r"(?:temperature|T\s*=|at|@)\s*(\d+\.?\d*)\s*K"
```

### Performance Issues

If enrichment is slow:

1. Reduce batch size in `ingest_enhanced.py`
2. Disable unused extraction patterns
3. Use multiprocessing for chunks
4. Consider streaming ingestion

## API Reference

```python
from metadata_enricher import MetadataEnricher

enricher = MetadataEnricher()

# Enrich a single chunk
metadata = enricher.enrich(
    text="...",
    doc_id=1,
    chunk_id=1,
    filename="paper.pdf",
    page_number=5,
    section="Results"
)
```

## Citation

If using this enhanced database in research:

```bibtex
@dataset{co2m_enhanced_2026,
  title={Enhanced CO2M Vector Database with Structured Metadata},
  year={2026},
  description={Semantic search database with auto-extracted material, measurement, and experimental metadata}
}
```
