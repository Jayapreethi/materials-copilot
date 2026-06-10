# materials-copilot

Materials Copilot is a modular scientific literature intelligence system composed of three interoperable applications: an offline PDF intelligence pipeline, a vector-based RAG retrieval service, and a chat-based research agent.

## CO2M Literature Intelligence System

One of the technical components of Materials Copilot is the **CO2M Literature Intelligence System** — an interactive knowledge-extraction and exploration platform for the CO₂ capture and mineralization literature corpus.

<img width="1642" height="913" alt="image" src="https://github.com/user-attachments/assets/c0b2b701-4643-48fc-8b9b-1f1caa2df708" />

It runs a 7-stage offline pipeline (taxonomy loading → metadata enrichment → PDF discovery → text extraction & chunking → concept classification → extractive summaries → visualization table generation) and materializes all outputs as Parquet data cubes. The dashboard reads only from these artifacts, making it safe for air-gapped and restricted-network compute environments.

**Corpus:** 47 PDFs · 3,689 chunks · 2,104 concept assignments

**Dashboard capabilities:** concept-filtered document ranking, temporal research trend visualization, keyword search, and source snippet retrieval with provenance tracing.

<img width="2880" height="1618" alt="image" src="https://github.com/user-attachments/assets/ae9c399c-eaa0-4466-9b94-e4e1f34eba9e" />

---

## Quickstart

```bash
git clone https://github.com/Jayapreethi/materials-copilot.git
cd materials-copilot
pip install -r requirements.txt

# Run the CO2M pipeline
python co2m_lit_dashboard/src/build_corpus.py --pdf-dir co2m/pdfs --metadata-dir co2m/metadata --out co2m/artifacts

# Launch the dashboard
streamlit run co2m_lit_dashboard/dashboard/app.py
```

For HPC (SLURM):

```bash
cd hpc_tools/ && sbatch submit_ingestion.sh
```

---

## Author

**Jaya Preethi Mohan**

Doctoral Researcher, Computer Science — University of North Dakota

Former Computer Scientist, Advanced Computing Team — Pacific Northwest National Laboratory

[GitHub](https://github.com/Jayapreethi) · [Portfolio](https://jayapreethi.github.io)

---

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
