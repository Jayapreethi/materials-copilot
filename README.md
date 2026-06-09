# materials-copilot

A RAG-powered copilot for materials science workflows.

## CO2M Literature Intelligence Dashboard

Interactive analysis platform for CO2M literature corpus with concept mapping, temporal trends, and document retrieval.

![Dashboard Overview](docs/dashboard-overview.png)

**Features:**
* Stacked area visualization of research activity trends across concepts
* Top documents ranked by relevance per concept
* Real-time filtering by scientific concept, year, and keywords
* Corpus statistics (47 PDFs, 3689 chunks, 2104 concept assignments)

## Structure

* `ingestion/` Data ingestion (papers, datasets, structures)
* `embeddings/` Embedding model code and utilities
* `vector_db/` Vector database setup and management
* `rag_pipeline/` Retrieval-augmented generation pipeline
* `hpc_tools/` HPC integration (SLURM, job submission, etc.)
* `ui/` User interface
* `configs/` Configuration files
* `examples/` Example scripts and notebooks
* `docs/` Documentation
* `co2m_lit_dashboard/` CO2M literature intelligence pipeline and dashboard
