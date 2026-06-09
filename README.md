# materials-copilot

A RAG-powered copilot for materials science workflows will have various features and tools for chem researchers. One of the tools in copilot if the CO2M literature Intelligence dashboard.

## CO2M Literature Intelligence Dashboard

Interactive analysis platform for CO2M literature corpus with concept mapping, temporal trends, and document retrieval.

![Dashboard Overview](docs/dashboard-overview.png)

**Features:**
* Stacked area visualization of research activity trends across concepts
* Top documents ranked by relevance per concept
* Real-time filtering by scientific concept, year, and keywords
* Corpus statistics (47 PDFs, 3689 chunks, 2104 concept assignments)

<img width="2880" height="1618" alt="image" src="https://github.com/user-attachments/assets/ae9c399c-eaa0-4466-9b94-e4e1f34eba9e" />


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
