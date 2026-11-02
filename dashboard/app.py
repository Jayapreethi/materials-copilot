"""
dashboard/app.py
CO2M Literature Intelligence Dashboard
Run: streamlit run co2m_lit_dashboard/dashboard/app.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to path for imports
_HERE = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# RAG Pipeline imports
# Keep these optional so the dashboard still loads even when a native dependency
# (for example on Windows) prevents the embedding stack from being imported.
try:
    from rag_pipeline.query_service import RAGQueryService
    _RAG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised in runtime fallback
    RAGQueryService = None
    _RAG_IMPORT_ERROR = exc

try:
    from rag_pipeline.verification import RAGVerification
    _VERIFICATION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional module
    RAGVerification = None
    _VERIFICATION_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "co2m" / "artifacts"

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Process Engineering AI Copilot: Lab Discovery to Pilot Scale Development",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_ACCENT  = "#1D4ED8"
_ACCENT2 = "#0284C7"
_MUTED   = "#6B7280"
_SURFACE = "#F9FAFB"
_BORDER  = "#E5E7EB"
_TEXT    = "#111827"
_GOOD    = "#065F46"
_FONT    = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: {_FONT} !important;
    color: {_TEXT};
}}
h1, h2, h3, h4 {{ font-family: {_FONT} !important; letter-spacing: -0.01em; }}

section[data-testid="stSidebar"] {{
    background: {_SURFACE};
    border-right: 1px solid {_BORDER};
}}

[data-testid="metric-container"] {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 0.9rem 1.1rem !important;
}}
[data-testid="metric-container"] label {{
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: {_MUTED} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: {_TEXT} !important;
}}

.co2m-summary {{
    border-left: 3px solid {_ACCENT};
    background: {_SURFACE};
    padding: 0.85rem 1.1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #1F2937;
    margin: 0.5rem 0 1.5rem 0;
}}

.co2m-snippet {{
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    background: #FFFFFF;
}}
.co2m-snippet .meta {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {_MUTED};
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 0.3rem;
}}
.co2m-snippet .body {{
    font-size: 0.875rem;
    color: #374151;
    line-height: 1.65;
}}
.co2m-snippet mark {{
    background: #FEF08A;
    color: {_TEXT};
    border-radius: 2px;
    padding: 0 2px;
}}

/* Streamlit tab strip */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 2px solid {_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: {_FONT} !important;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.55rem 1.1rem;
    color: {_MUTED};
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}}
.stTabs [aria-selected="true"] {{
    color: {_ACCENT} !important;
    border-bottom: 2px solid {_ACCENT} !important;
    font-weight: 600 !important;
}}

hr {{ border: none; border-top: 1px solid {_BORDER}; margin: 1.25rem 0; }}
.stDataFrame {{ border: 1px solid {_BORDER}; border-radius: 6px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Layout helper for Plotly (no duplicate key errors)
# ---------------------------------------------------------------------------
def _layout(**extra) -> dict:
    base = dict(
        font=dict(family=_FONT, size=12, color=_TEXT),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=36, b=10),
        title_font=dict(family=_FONT, size=13, color=_MUTED),
        title_x=0,
        xaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        hoverlabel=dict(font_family=_FONT),
    )
    for axis_key in ("xaxis", "yaxis"):
        if axis_key in extra:
            base[axis_key] = {**base[axis_key], **extra.pop(axis_key)}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading artifacts …")
def load_all():
    names = [
        "documents", "chunks", "concepts", "summaries",
        "concept_counts", "top_documents_by_concept", "keywords_by_concept",
    ]
    data = {n: pd.read_parquet(ARTIFACTS_DIR / f"{n}.parquet")
            if (ARTIFACTS_DIR / f"{n}.parquet").exists()
            else pd.DataFrame()
            for n in names}
    manifest: dict = {}
    mp = ARTIFACTS_DIR / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    return data, manifest


if not ARTIFACTS_DIR.exists():
    st.warning(
        f"**Dashboard artifacts not found:** `{ARTIFACTS_DIR}`. "
        "Artifact-based tabs are disabled, but Semantic Search uses the live Postgres vector database."
    )

data, manifest = load_all()
documents_df     = data["documents"]
chunks_df        = data["chunks"]
concepts_df      = data["concepts"]
summaries_df     = data["summaries"]
concept_counts_df= data["concept_counts"]
top_docs_df      = data["top_documents_by_concept"]
keywords_df      = data["keywords_by_concept"]
has_artifacts = not concept_counts_df.empty or not summaries_df.empty

has_year = (
    not documents_df.empty
    and "year" in documents_df.columns
    and documents_df["year"].notna().any()
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
hcol, dcol = st.columns([7, 1])
with hcol:
    st.title("Process Engineering AI Copilot: Lab Discovery to Pilot Scale Development")
with dcol:
    if manifest:
        ts = manifest.get("build_timestamp", "")
        if ts:
            st.caption(f"Built {ts[:10]}")
st.markdown("---")

# ---------------------------------------------------------------------------
# Live Postgres corpus stats (used when artifacts are unavailable)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def live_corpus_stats() -> dict:
    try:
        from vector_db.postgres_corpus_store import PostgresCorpusStore
        store = PostgresCorpusStore()
        return store.stats()
    except Exception:
        return {"documents": 0, "chunks": 0, "embedded_chunks": 0, "unique_documents": 0}

live_stats = live_corpus_stats()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Filters")

all_concepts: list[str] = (
    concept_counts_df["concept_name"].tolist()
    if not concept_counts_df.empty and "concept_name" in concept_counts_df.columns
    else summaries_df["concept_name"].tolist()
    if not summaries_df.empty
    else []
)

# Fallback: derive concept/topic options from live Postgres metadata when artifacts are absent.
if not all_concepts:
    try:
        from vector_db.postgres_corpus_store import PostgresCorpusStore
        _concept_store = PostgresCorpusStore()
        with _concept_store.connect() as _conn:
            _rows = _conn.execute(
                "SELECT DISTINCT kw FROM documents, LATERAL ("
                "SELECT jsonb_array_elements_text(metadata->'keywords') AS kw WHERE jsonb_typeof(metadata->'keywords') = 'array'"
                " UNION ALL SELECT metadata->>'keywords' AS kw WHERE jsonb_typeof(metadata->'keywords') = 'string'"
                ") kws WHERE kw IS NOT NULL AND kw != ''"
            ).fetchall()
        live_concepts = sorted({row["kw"] for row in _rows if row.get("kw")})
        standard_topics = [
            # Carbon management
            "CO2 Capture", "CO2 Conversion", "CO2 Storage", "CO2 Utilization",
            "Carbon Mineralization", "Carbonation", "Direct Air Capture", "Ocean CDR",
            "Point Source Capture", "Post-Combustion Capture", "Pre-Combustion Capture",
            "Oxy-Fuel Combustion", "Chemical Looping", "Calcium Looping", "Carbonate Looping",
            "Amine Scrubbing", "Solid Sorbents", "Membrane Separation", "Cryogenic Separation",
            "Pressure Swing Adsorption", "Temperature Swing Adsorption", "Vacuum Swing Adsorption",
            "Electrochemical CO2 Reduction", "Photocatalytic CO2 Reduction", "Biocatalytic CO2 Fixation",
            "CO2 to Fuels", "CO2 to Chemicals", "CO2 to Materials", "Methanation",
            "Fischer-Tropsch Synthesis", "Methanol Synthesis", "Formic Acid Production",
            # Hydrogen
            "Hydrogen Production", "Hydrogen Storage", "Hydrogen Transport", "Hydrogen Utilization",
            "Water Electrolysis", "Alkaline Electrolysis", "PEM Electrolysis", "Solid Oxide Electrolysis",
            "Photoelectrochemical Water Splitting", "Photocatalytic Water Splitting",
            "Thermochemical Hydrogen Production", "Methane Pyrolysis", "Steam Methane Reforming",
            "Autothermal Reforming", "Biomass Gasification", "Coal Gasification",
            "Hydrogen Compression", "Hydrogen Liquefaction", "Liquid Organic Hydrogen Carriers",
            "Ammonia as Hydrogen Carrier", "Metal Hydrides", "Compressed Hydrogen Storage",
            # Energy storage
            "Lithium-Ion Batteries", "Solid-State Batteries", "Flow Batteries", "Sodium-Ion Batteries",
            "Battery Manufacturing", "Battery Recycling", "Thermal Energy Storage", "Sensible Heat Storage",
            "Latent Heat Storage", "Thermochemical Storage", "Compressed Air Energy Storage",
            "Pumped Hydro Storage", "Flywheel Energy Storage", "Supercapacitors", "Hydrogen Storage",
            # Process engineering
            "Process Intensification", "Scale-Up", "Pilot Plant", "Bench Scale", "Laboratory Scale",
            "Continuous Flow Reactors", "Batch Reactors", "Packed Bed Reactors", "Fluidized Bed Reactors",
            "Bubble Column Reactors", "Slurry Reactors", "Fixed Bed Reactors", "Membrane Reactors",
            "Reactive Distillation", "Heat Exchangers", "Separations", "Distillation", "Extraction",
            "Crystallization", "Precipitation", "Filtration", "Centrifugation", "Drying",
            "Process Control", "Process Optimization", "Process Simulation", "Process Design",
            "Techno-Economic Analysis", "Life Cycle Assessment", "Pinch Analysis", "Exergy Analysis",
            "Energy Integration", "Waste Heat Recovery", "Industrial Decarbonization", "Electrification",
            # Materials
            "Materials Synthesis", "Materials Characterization", "Catalysts", "Sorbents", "Membranes",
            "Metal-Organic Frameworks", "Zeolites", "Porous Materials", "Nanomaterials", "Polymers",
            "Ionic Liquids", "Deep Eutectic Solvents", "Carbon Materials", "Metal Oxides",
            "Perovskites", "Single-Atom Catalysts", "Heterogeneous Catalysis", "Homogeneous Catalysis",
            "Electrocatalysts", "Photocatalysts", "Biocatalysts", "Enzyme Catalysis",
            # Analysis methods
            "Techno-Economic Analysis", "Life Cycle Assessment", "Energy Efficiency",
            "Thermodynamics", "Kinetics", "Mass Transfer", "Heat Transfer", "Fluid Dynamics",
            "Reaction Engineering", "Transport Phenomena", "Process Modeling", "CFD Simulation",
            "Molecular Simulation", "Density Functional Theory", "Machine Learning for Materials",
            "High-Throughput Screening", "Experimental Design", "Statistical Analysis",
        ]
        all_concepts = sorted(set(live_concepts + standard_topics))[:500]
    except Exception:
        all_concepts = []

if not all_concepts:
    st.sidebar.caption("No concept data available; semantic search remains available.")
    selected_concept = "No concept data"
    year_filter = []
    keyword_search = ""
else:
    selected_concept: str = st.sidebar.selectbox("Scientific Concept", all_concepts)

    year_filter: list[int] = []
    years: list[int] = []
    if has_year:
        years = sorted(documents_df["year"].dropna().astype(int).unique().tolist())
    elif all_concepts:
        try:
            from vector_db.postgres_corpus_store import PostgresCorpusStore
            _year_store = PostgresCorpusStore()
            with _year_store.connect() as _conn:
                _years = _conn.execute(
                    "SELECT DISTINCT publication_year FROM documents WHERE publication_year IS NOT NULL ORDER BY publication_year"
                ).fetchall()
            years = [int(row["publication_year"]) for row in _years if row.get("publication_year")]
        except Exception:
            years = []
    if len(years) > 1:
        year_filter = st.sidebar.multiselect("Year", options=years, default=[])

    keyword_search: str = st.sidebar.text_input("Snippet keyword", placeholder="filter snippets …")

st.sidebar.markdown("---")
if manifest:
    st.sidebar.markdown("**Corpus**")
    st.sidebar.markdown(
        f"PDFs: **{live_stats['documents']:,}**  \n"
        f"Chunks: **{live_stats['chunks']:,}**  \n"
        f"Unique Docs: **{live_stats['unique_documents']:,}**"
    )

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
concept_rows = (
    concepts_df[concepts_df["concept_name"] == selected_concept].copy()
    if not concepts_df.empty and selected_concept != "No concept data" else pd.DataFrame()
)
concept_chunk_ids: set[int] = set(concept_rows["chunk_id"].unique()) if not concept_rows.empty else set()
concept_doc_ids: set[int]   = set(concept_rows["doc_id"].unique())   if not concept_rows.empty else set()

if year_filter and has_year and not documents_df.empty:
    year_doc_ids = set(documents_df.loc[documents_df["year"].isin(year_filter), "doc_id"])
    concept_doc_ids &= year_doc_ids
    concept_rows = concept_rows[concept_rows["doc_id"].isin(concept_doc_ids)]
    concept_chunk_ids = set(concept_rows["chunk_id"].unique())

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
total_pdfs = live_stats["documents"] if live_stats["documents"] else (len(documents_df) if not documents_df.empty else 0)
total_chunks = live_stats["chunks"] if live_stats["chunks"] else (len(chunks_df) if not chunks_df.empty else 0)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Documents", f"{total_pdfs:,}")
m2.metric("Unique Documents", f"{live_stats['unique_documents']:,}")
m3.metric("Total Chunks", f"{total_chunks:,}")
if has_year and concept_doc_ids and not documents_df.empty:
    yr_s = documents_df.loc[documents_df["doc_id"].isin(concept_doc_ids), "year"].dropna()
    m4.metric("Peak Year", str(int(yr_s.mode().iloc[0])) if not yr_s.empty else "—")
else:
    m4.metric("Embedded Chunks", f"{live_stats['embedded_chunks']:,}" if live_stats['embedded_chunks'] else "0")

# ---------------------------------------------------------------------------
# Concept summary
# ---------------------------------------------------------------------------
if selected_concept != "No concept data":
    st.markdown(f"#### {selected_concept}")
    summary_text = "No summary available for this concept."
    if not summaries_df.empty:
        _r = summaries_df[summaries_df["concept_name"] == selected_concept]
        if not _r.empty:
            summary_text = _r.iloc[0]["summary_text"]
    st.markdown(f'<div class="co2m-summary">{summary_text}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live corpus categorization
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def live_corpus_categories() -> dict[str, list[str]]:
    try:
        from vector_db.postgres_corpus_store import PostgresCorpusStore
        store = PostgresCorpusStore()
        with store.connect() as conn:
            rows = conn.execute("""
                SELECT DISTINCT jsonb_array_elements_text(metadata->'material_classes') AS category
                FROM documents WHERE metadata ? 'material_classes'
                UNION
                SELECT DISTINCT jsonb_array_elements_text(metadata->'co2_processes') AS category
                FROM documents WHERE metadata ? 'co2_processes'
                UNION
                SELECT DISTINCT jsonb_array_elements_text(metadata->'measurement_methods') AS category
                FROM documents WHERE metadata ? 'measurement_methods'
                UNION
                SELECT DISTINCT jsonb_array_elements_text(metadata->'reported_properties') AS category
                FROM documents WHERE metadata ? 'reported_properties'
                ORDER BY category
            """).fetchall()
        live = [r["category"] for r in rows if r.get("category")]
        standard = [
            "CO2 Capture", "CO2 Conversion", "CO2 Storage", "Hydrogen Production",
            "Hydrogen Storage", "Electrolysis", "Catalysis", "Membrane Separation",
            "Adsorption", "Absorption", "Mineralization", "Process Intensification",
            "Scale-Up", "Pilot Plant", "Bench Scale", "Laboratory Scale",
            "Techno-Economic Analysis", "Life Cycle Assessment", "Energy Efficiency",
            "Materials Synthesis", "Characterization", "Thermodynamics", "Kinetics",
            "Sorbents", "Catalysts", "Membranes", "MOFs", "Zeolites", "Amines",
        ]
        return {"Material Classes": sorted(set(live + standard))}
    except Exception:
        return {"Material Classes": []}

live_categories = live_corpus_categories()
if live_categories:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Corpus Categorization")
    category_group = st.sidebar.selectbox("Category Group", options=list(live_categories.keys()))
    if category_group and live_categories.get(category_group):
        selected_category = st.sidebar.selectbox("Category", options=live_categories[category_group])

# ===========================================================================
# TABBED NAVIGATION
# ===========================================================================
tab_rag, tab_overview, tab_science, tab_deepdive, tab_domain, tab_compare, tab_docs, tab_snippets, tab_verify = st.tabs([
    "Semantic Search",
    "Corpus Overview",
    "Scientific Insights",
    "Concept Deep Dive",
    "Domain Map",
    "Cross-Concept",
    "Documents",
    "Source Snippets",
    "System Verification",
])

# ============================================================
# TAB 0 — Semantic Search
# ============================================================
with tab_rag:
    st.markdown("### Semantic Search")

    if RAGQueryService is None:
        st.warning("Semantic search is unavailable because the RAG dependencies could not be loaded.")
        if _RAG_IMPORT_ERROR is not None:
            st.code(str(_RAG_IMPORT_ERROR))
        st.info("This can happen on Windows when a native dependency such as a DLL is missing. Install or repair the embedding stack and reload the app.")
        st.stop()

    # Initialize service (cached)
    @st.cache_resource
    def get_query_service():
        """Load query service once."""
        try:
            service = RAGQueryService(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                vector_store_path=str(PROJECT_ROOT / "vector_db" / "data"),
                collection_name="co2m_corpus",
                device="cpu",
            )
            return service
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.info("Make sure the vector database is populated. Run: `python ingest_enhanced.py`")
            raise

    try:
        query_service = get_query_service()
        stats = query_service.get_stats()
        st.info(f"Vector DB: {stats.get('total_documents', 0):,} documents indexed")
    except Exception:
        st.stop()

    # Check LLM status
    llm_status = query_service.get_llm_status()
    
    # Search input
    query = st.text_input(
        "Enter your question or search term:",
        placeholder="e.g., 'CO2 capture materials', 'amine-based sorbents', 'separation efficiency'",
        key="rag_query",
    )

    # Search parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Results:", min_value=1, max_value=20, value=5, key="top_k_rag")
    with col2:
        include_context = st.checkbox("Context summary", value=True, key="include_context_rag")
    with col3:
        concept_filter = st.selectbox(
            "Concept filter:",
            options=["None"] + all_concepts,
            key="concept_filter_rag",
        )
    
    # LLM generation option
    col_llm1, col_llm2 = st.columns(2)
    with col_llm1:
        use_llm = st.checkbox(
            "Generate with AI",
            value=True,
            key="use_llm",
            help="Use AI to generate a comprehensive answer based on retrieved documents"
        )
    
    with col_llm2:
        if llm_status.get("available"):
            provider = llm_status.get('provider', 'unknown').title()
            if llm_status.get('auto_detected'):
                st.caption(f"âœ“ Auto-detected: {provider}")
            else:
                st.caption(f"âœ“ {provider}")
        else:
            st.caption("âš ï¸ AI not available. Install Ollama or set API keys for free/paid LLM access")
    
    trl_filter = st.selectbox(
        "Scale-derived TRL proxy:",
        options=["None", "TRL_2_3_PROXY", "TRL_4_5_PROXY", "TRL_6_7_PROXY", "INSUFFICIENT_EVIDENCE", "REVIEW_EXCLUDED"],
        key="trl_proxy_filter_rag",
        help="Filter retrieved chunks by the two-pass scale-derived TRL proxy metadata",
    )
    if trl_filter == "None":
        trl_filter = None
    
    if concept_filter == "None":
        concept_filter = None

    # Initialize session state for search results
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = None

    # Execute search
    search_button_col, clear_button_col = st.columns([4, 1])
    with search_button_col:
        search_clicked = st.button("Search", key="search_button", use_container_width=True)
    with clear_button_col:
        clear_clicked = st.button("Clear", key="clear_button", use_container_width=True)
    
    if clear_clicked:
        st.session_state.search_results = None
        st.session_state.search_query = None
        st.rerun()
    
    if query and search_clicked:
        try:
            search_msg = "Searching with AI..." if llm_status.get("available") else "Searching semantically..."
            with st.spinner(search_msg):
                if llm_status.get("available"):
                    results = query_service.generate_with_llm(
                        question=query,
                        top_k=top_k,
                        concept_filter=concept_filter,
                        temperature=0.7,
                        max_tokens=500,
                        trl_proxy=trl_filter,
                    )
                else:
                    results = query_service.query(
                        question=query,
                        top_k=top_k,
                        concept_filter=concept_filter,
                        include_context=include_context,
                        trl_proxy=trl_filter,
                    )
            # Store results in session state
            st.session_state.search_results = results
            st.session_state.search_query = query
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            logger.error(f"Search error: {e}", exc_info=True)
            st.session_state.search_results = None
            st.stop()
    
    # Display stored results if available
    if st.session_state.search_results:
        results = st.session_state.search_results

        # Display LLM response if available (regardless of current checkbox state, since results may have been generated with LLM)
        if "llm_response" in results and results.get("llm_response"):
            st.markdown("---")
            st.markdown("### 🤖 AI-Generated Answer")
            
            structured = results.get("structured_answer")
            if isinstance(structured, dict) and structured.get("status"):
                status_label = structured["status"].replace("_", " ").title()
                st.caption(f"Status: {status_label}")
                st.markdown(structured.get("direct_answer") or "")
                findings = structured.get("findings") or []
                if findings:
                    st.markdown("#### Findings")
                    for idx, finding in enumerate(findings, 1):
                        st.markdown(f"**{idx}. {finding.get('statement', '')}**")
                        if finding.get("material_or_technology"):
                            st.caption(f"Material/Technology: {finding['material_or_technology']}")
                        if finding.get("quantitative_result"):
                            st.caption(f"Quantitative result: {finding['quantitative_result']}")
                        if finding.get("reported_conditions"):
                            st.caption(f"Conditions: {finding['reported_conditions']}")
                        if finding.get("limitation"):
                            st.caption(f"Limitation: {finding['limitation']}")
                        evidence = finding.get("evidence") or []
                        if evidence:
                            with st.expander(f"Evidence for finding {idx}", expanded=False):
                                for item in evidence:
                                    st.caption(f"[{item.get('source_id')}] {item.get('exact_quote', '')}")
                if structured.get("synthesis"):
                    st.markdown("#### Synthesis")
                    st.markdown(structured["synthesis"])
                gaps = structured.get("evidence_gaps") or []
                if gaps:
                    with st.expander("Evidence gaps", expanded=False):
                        for gap in gaps:
                            st.caption(f"- {gap}")
                if structured.get("expanded_search_required"):
                    st.info("Additional retrieval is recommended for a broader answer.")
            else:
                try:
                    st.markdown(results['llm_response'])
                except Exception:
                    st.text(results['llm_response'])
            
            # Show LLM metadata in a collapsible section
            with st.expander("Response Details", expanded=False):
                info_cols = st.columns(3)
                with info_cols[0]:
                    provider = results.get("llm_provider", "N/A")
                    st.metric("Provider", provider.title())
                with info_cols[1]:
                    model = results.get("llm_model", "N/A")
                    st.metric("Model", model.split("/")[-1][:20])  # Shorten long model names
                with info_cols[2]:
                    tokens = results.get("llm_usage", {}).get("total_tokens", results.get("llm_usage", {}).get("tokens", 0))
                    st.metric("Tokens", tokens if tokens else "N/A")
            
            # Show sources used
            if results.get("llm_sources"):
                with st.expander("Sources Used in Answer", expanded=False):
                    for i, source in enumerate(results["llm_sources"][:5], 1):
                        file = (
                            source.get("file")
                            or source.get("filename")
                            or source.get("document_title")
                            or "Unknown"
                        )
                        page = (
                            source.get("page")
                            or source.get("page_number")
                            or source.get("page_start")
                            or "N/A"
                        )
                        sim = source.get("similarity", 0)
                        st.caption(f"{i}. {file} (Page {page}) — {sim:.1%} match")
            
            st.markdown("---")

        # Display results summary
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("ðŸ“„ Documents Found", results.get('num_results', 0))
        with col_r2:
            summary = results.get("context_summary", {})
            st.metric("ðŸ“Š Unique Sources", summary.get('unique_documents', 0))
        with col_r3:
            avg_sim = summary.get('avg_similarity', 0)
            st.metric("ðŸ“ˆ Match Score", f"{avg_sim:.1%}")

        # Display summary if context was generated
        if "context_summary" in results and results.get("context_summary"):
            summary = results["context_summary"]
            with st.expander("Summary Statistics", expanded=True):
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1:
                    st.metric("Results", summary.get("total_results", 0))
                with scol2:
                    st.metric("Unique Docs", summary.get("unique_documents", 0))
                with scol3:
                    st.metric("Avg Similarity", f"{summary.get('avg_similarity', 0):.3f}")
                with scol4:
                    st.metric("Materials", summary.get("materials_found", 0))

                if "co2_uptake_stats" in summary and summary["co2_uptake_stats"]:
                    co2_stats = summary["co2_uptake_stats"]
                    st.markdown("**COâ‚‚ Uptake (mmol/g)**")
                    co2col1, co2col2, co2col3 = st.columns(3)
                    with co2col1:
                        st.metric("Avg", f"{co2_stats.get('avg', 0):.2f}")
                    with co2col2:
                        st.metric("Min", f"{co2_stats.get('min', 0):.2f}")
                    with co2col3:
                        st.metric("Max", f"{co2_stats.get('max', 0):.2f}")

        # Display visualizations
        result_list = results.get("results", [])
        if result_list:
            st.markdown("### Result Visualizations")
            
            viz_cols = st.columns(2)
            
            # Plot 1: Similarity Distribution
            with viz_cols[0]:
                similarity_data = [r.get("similarity", 0) for r in result_list]
                ranks = [r.get("rank", i+1) for i, r in enumerate(result_list)]
                
                fig_sim = px.bar(
                    x=ranks,
                    y=similarity_data,
                    labels={"x": "Result Rank", "y": "Similarity Score"},
                    title="Semantic Similarity by Result",
                    color=similarity_data,
                    color_continuous_scale="Viridis"
                )
                fig_sim.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)
            
            # Plot 2: Source Distribution
            with viz_cols[1]:
                source_counts = {}
                for r in result_list:
                    fname = r.get("filename", "Unknown")
                    source_counts[fname] = source_counts.get(fname, 0) + 1
                
                fig_src = px.pie(
                    names=list(source_counts.keys()),
                    values=list(source_counts.values()),
                    title="Results by Source Document"
                )
                fig_src.update_layout(height=300)
                st.plotly_chart(fig_src, use_container_width=True)

        # Display context
        if "context" in results and results.get("context"):
            with st.expander("Formatted Context", expanded=False):
                st.text_area(
                    "Context",
                    value=results["context"],
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                )

        # Display individual results
        if result_list:
            st.markdown("#### Detailed Results")
            for result in result_list:
                with st.container():
                    rcol1, rcol2 = st.columns([1, 4])
                    with rcol1:
                        similarity_pct = result.get("similarity", 0) * 100
                        st.metric(f"#{result.get('rank', '?')}", f"{similarity_pct:.0f}%")
                    with rcol2:
                        st.markdown(
                            f"**{result.get('filename', 'Unknown')}** (Page {result.get('page_number', '?')})\n\n"
                            f"*{result.get('text', '')[:300]}...*"
                        )
                        if result.get("material_name"):
                            st.caption(f"Material: {result['material_name']}")
                        if result.get("trl_proxy"):
                            st.caption(f"Scale-derived TRL proxy: {result['trl_proxy']}")
                        if result.get("co2_uptake_mmol_g"):
                            st.caption(f"COâ‚‚ Uptake: {result['co2_uptake_mmol_g']} mmol/g")
                    st.divider()


    # Example queries
    with st.expander("Example Queries", expanded=False):
        st.markdown(
            """
            - What materials capture CO2 efficiently?
            - Compare amine and metal-organic frameworks
            - What are recent advances in sorption?
            - How does CO2 uptake vary by material?
            - What is the role of surface area in capture?
            """
        )

# ============================================================
# TAB 1 — Corpus Overview
# ============================================================
with tab_overview:
    st.markdown("#### Concept Coverage")
    ov1, ov2 = st.columns(2)

    with ov1:
        if not concept_counts_df.empty:
            df_p = concept_counts_df.sort_values("chunk_count", ascending=True).copy()
            df_p["_s"] = df_p["concept_name"].apply(lambda c: "Selected" if c == selected_concept else "Other")
            fig = px.bar(df_p, x="chunk_count", y="concept_name", orientation="h",
                         color="_s", color_discrete_map={"Selected": _ACCENT, "Other": "#D1D5DB"},
                         labels={"chunk_count": "Chunks", "concept_name": ""},
                         title="Coverage by chunks")
            fig.update_traces(showlegend=False)
            fig.update_layout(**_layout(height=360))
            st.plotly_chart(fig, use_container_width=True)

    with ov2:
        if not concept_counts_df.empty:
            df_d = concept_counts_df.sort_values("doc_count", ascending=True).copy()
            fig = px.bar(df_d, x="doc_count", y="concept_name", orientation="h",
                         color="doc_count",
                         color_continuous_scale=[[0, "#BFDBFE"], [1, _ACCENT]],
                         labels={"doc_count": "Documents", "concept_name": ""},
                         title="Coverage by documents")
            fig.update_layout(**_layout(height=360, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    if not concepts_df.empty:
        st.markdown("#### Concept Co-occurrence")
        st.caption("Number of documents shared between each pair of concepts.")
        dc_uniq = concepts_df[["doc_id", "concept_name"]].drop_duplicates()
        clist = sorted(dc_uniq["concept_name"].unique())
        comat = pd.DataFrame(0, index=clist, columns=clist)
        for _, grp in dc_uniq.groupby("doc_id"):
            names = grp["concept_name"].tolist()
            for i, a in enumerate(names):
                for b in names[i:]:
                    comat.loc[a, b] += 1
                    comat.loc[b, a] += 1
        fig = px.imshow(comat, color_continuous_scale="Blues",
                        labels=dict(color="Docs"),
                        title="Co-occurrence (shared documents)", aspect="auto")
        fig.update_layout(**_layout(height=430, margin=dict(l=10, r=10, t=40, b=10)))
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

    if has_year and not concepts_df.empty and not documents_df.empty:
        st.markdown("#### Activity Over Time")
        yc = (concepts_df[["doc_id", "concept_name"]].drop_duplicates()
              .merge(documents_df[["doc_id", "year"]].dropna(subset=["year"]),
                     on="doc_id", how="inner"))
        if not yc.empty:
            yc["year"] = yc["year"].astype(int)
            ycp = (yc.groupby(["year", "concept_name"]).size()
                   .reset_index(name="docs")
                   .pivot(index="concept_name", columns="year", values="docs")
                   .fillna(0))
            fig = px.imshow(ycp, color_continuous_scale="Blues",
                            labels=dict(x="Year", y="", color="Docs"),
                            title="Documents per concept per year", aspect="auto")
            fig.update_layout(**_layout(height=370, margin=dict(l=10, r=10, t=40, b=10)))
            fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
            fig.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — Scientific Insights  (all concepts at once)
# ============================================================
with tab_science:
    st.markdown("#### Scientific Insights — All Concepts")
    st.caption(
        "A holistic view of every concept in the corpus. "
        "Scroll through each panel to understand the full research landscape."
    )

    # â”€â”€ A. Summary cards: doc count + chunk count per concept â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not summaries_df.empty and not concept_counts_df.empty:
        st.markdown("##### Concept Summaries")
        merged_summary = summaries_df.merge(
            concept_counts_df[["concept_name", "doc_count", "chunk_count"]],
            on="concept_name", how="left",
        )
        for _, sr in merged_summary.iterrows():
            is_sel = sr["concept_name"] == selected_concept
            border_color = _ACCENT if is_sel else _BORDER
            bg = "#EFF6FF" if is_sel else "#FFFFFF"
            doc_c  = int(sr["doc_count"])   if pd.notna(sr.get("doc_count"))   else 0
            chk_c  = int(sr["chunk_count"]) if pd.notna(sr.get("chunk_count")) else 0
            st.markdown(
                f'<div style="border:1px solid {border_color};background:{bg};'
                f'border-radius:7px;padding:.75rem 1rem;margin-bottom:.5rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
                f'<span style="font-weight:600;font-size:.95rem;color:{_ACCENT if is_sel else _TEXT}">{sr["concept_name"]}</span>'
                f'<span style="font-size:.75rem;color:{_MUTED}">{doc_c} docs &nbsp;Â·&nbsp; {chk_c:,} chunks</span>'
                f'</div>'
                f'<div style="font-size:.85rem;color:#374151;line-height:1.6;margin-top:.35rem">{sr["summary_text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # â”€â”€ B. Side-by-side concept size comparison (treemap) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not concept_counts_df.empty:
        st.markdown("##### Research Volume Treemap")
        st.caption("Area is proportional to total chunks assigned to each concept.")
        fig_tm = px.treemap(
            concept_counts_df,
            path=["concept_name"],
            values="chunk_count",
            color="doc_count",
            color_continuous_scale=[[0, "#BFDBFE"], [0.5, _ACCENT2], [1, _ACCENT]],
            labels={"concept_name": "Concept", "chunk_count": "Chunks", "doc_count": "Docs"},
            title="Research volume per concept (chunks)",
        )
        fig_tm.update_traces(
            textinfo="label+value",
            textfont=dict(family=_FONT, size=12),
            hovertemplate="<b>%{label}</b><br>Chunks: %{value}<br>Docs: %{color:.0f}<extra></extra>",
        )
        fig_tm.update_layout(
            font=dict(family=_FONT, size=12, color=_TEXT),
            paper_bgcolor="#FFFFFF",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            title_font=dict(family=_FONT, size=13, color=_MUTED),
            title_x=0,
            coloraxis_colorbar=dict(title="Docs", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_tm, use_container_width=True)

    # â”€â”€ C. All-concept keyword heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not keywords_df.empty:
        st.markdown("##### Keyword Presence Across All Concepts")
        st.caption(
            "Top 15 keywords (by total TF-IDF weight across corpus) Ã— all concepts. "
            "Brighter = stronger association."
        )
        top15_kws = (
            keywords_df.groupby("keyword")["weight"]
            .sum().nlargest(15).index.tolist()
        )
        kw_heat = (
            keywords_df[keywords_df["keyword"].isin(top15_kws)]
            .pivot_table(index="keyword", columns="concept_name",
                         values="weight", aggfunc="mean")
            .fillna(0)
        )
        fig_kh = px.imshow(
            kw_heat,
            color_continuous_scale=[[0, "#F1F5F9"], [0.4, "#93C5FD"], [1, _ACCENT]],
            labels=dict(x="Concept", y="Keyword", color="TF-IDF"),
            title="Top keywords across all concepts",
            aspect="auto",
            text_auto=".3f",
        )
        fig_kh.update_layout(
            font=dict(family=_FONT, size=11, color=_TEXT),
            paper_bgcolor="#FFFFFF",
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
            title_font=dict(family=_FONT, size=13, color=_MUTED),
            title_x=0,
        )
        fig_kh.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        fig_kh.update_yaxes(tickfont=dict(size=10))
        fig_kh.update_traces(textfont=dict(size=8))
        st.plotly_chart(fig_kh, use_container_width=True)

    # â”€â”€ D. All-concept year trend (area) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if has_year and not concepts_df.empty and not documents_df.empty:
        st.markdown("##### Research Activity Over Time — All Concepts")
        st.caption("Stacked area shows the total chunk volume per year, coloured by concept.")
        yc_all = (
            concepts_df[["doc_id", "concept_name", "chunk_id"]]
            .drop_duplicates(subset=["doc_id", "concept_name", "chunk_id"])
            .merge(
                documents_df[["doc_id", "year"]].dropna(subset=["year"]),
                on="doc_id", how="inner",
            )
        )
        if not yc_all.empty:
            yc_all["year"] = yc_all["year"].astype(int)
            yc_agg = (
                yc_all.groupby(["year", "concept_name"])["chunk_id"]
                .count()
                .reset_index(name="chunks")
                .sort_values("year")
            )
            fig_area = px.area(
                yc_agg,
                x="year", y="chunks",
                color="concept_name",
                line_group="concept_name",
                labels={"chunks": "Chunks", "year": "Year", "concept_name": "Concept"},
                title="Research activity (all concepts) — stacked area",
            )
            fig_area.update_layout(
                font=dict(family=_FONT, size=12, color=_TEXT),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=380,
                margin=dict(l=10, r=10, t=44, b=60),
                title_font=dict(family=_FONT, size=13, color=_MUTED),
                title_x=0,
                xaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=False,
                           tickmode="linear", dtick=2, tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=False,
                           tickfont=dict(size=10)),
                legend=dict(
                    title=None,
                    orientation="h",
                    yanchor="top", y=-0.2,
                    xanchor="left", x=0,
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_area, use_container_width=True)

    # â”€â”€ E. Top 5 docs per concept (grouped bar) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not top_docs_df.empty and not documents_df.empty:
        st.markdown("##### Top Document per Concept")
        st.caption("The single best-matching document (most chunks) for each concept.")
        lc2 = "title" if "title" in top_docs_df.columns else "filename"
        top1 = (
            top_docs_df[top_docs_df["rank"] == 1][["concept_name", "chunk_count", lc2, "filename"]]
            .copy()
        )
        top1["label"] = top1[lc2].fillna(top1["filename"]).astype(str).str[:48]
        top1 = top1.sort_values("chunk_count", ascending=False)
        fig_top1 = px.bar(
            top1,
            x="chunk_count",
            y="concept_name",
            orientation="h",
            color="concept_name",
            text="label",
            labels={"chunk_count": "Chunks", "concept_name": ""},
            title="Best-matched document per concept",
        )
        fig_top1.update_traces(
            showlegend=False,
            textposition="inside",
            textfont=dict(size=9, color="#FFFFFF"),
            insidetextanchor="start",
        )
        fig_top1.update_layout(**_layout(height=360))
        st.plotly_chart(fig_top1, use_container_width=True)

# ============================================================
# TAB 3 — Concept Deep Dive
# ============================================================
with tab_deepdive:
    st.markdown(f"#### {selected_concept} — Analysis")
    dd1, dd2 = st.columns(2)

    with dd1:
        if not top_docs_df.empty:
            ct = top_docs_df[top_docs_df["concept_name"] == selected_concept].head(10).copy()
            if not ct.empty:
                lc = "title" if "title" in ct.columns else "filename"
                ct["label"] = ct[lc].fillna(ct.get("filename", "")).astype(str).str[:52]
                fig = px.bar(ct.sort_values("chunk_count", ascending=True),
                             x="chunk_count", y="label", orientation="h",
                             color="chunk_count",
                             color_continuous_scale=[[0, "#BFDBFE"], [1, _ACCENT]],
                             labels={"chunk_count": "Chunks", "label": ""},
                             title="Top documents")
                fig.update_layout(**_layout(height=370, coloraxis_showscale=False))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No documents matched this concept.")

    with dd2:
        if not concept_rows.empty and "confidence_score" in concept_rows.columns:
            fig = px.histogram(concept_rows, x="confidence_score", nbins=20,
                               color_discrete_sequence=[_ACCENT2],
                               labels={"confidence_score": "Confidence score"},
                               title="Confidence score distribution")
            fig.update_layout(**_layout(height=370))
            fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1)
            st.plotly_chart(fig, use_container_width=True)

    if not keywords_df.empty:
        kw = keywords_df[keywords_df["concept_name"] == selected_concept].head(20)
        if not kw.empty:
            st.markdown("#### Top TF-IDF Keywords")
            fig = px.bar(kw.sort_values("weight", ascending=True),
                         x="weight", y="keyword", orientation="h",
                         color="weight",
                         color_continuous_scale=[[0, "#D1FAE5"], [1, _GOOD]],
                         labels={"weight": "TF-IDF weight", "keyword": ""},
                         title=f"Keywords — {selected_concept}")
            fig.update_layout(**_layout(height=430, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    if has_year and not documents_df.empty and concept_doc_ids:
        yd = documents_df[
            documents_df["doc_id"].isin(concept_doc_ids) & documents_df["year"].notna()
        ].copy()
        if not yd.empty:
            st.markdown("#### Publications by Year")
            yc2 = (yd.groupby(yd["year"].astype(int)).size()
                   .reset_index(name="Documents")
                   .rename(columns={"year": "Year"})
                   .sort_values("Year"))
            fig = px.bar(yc2, x="Year", y="Documents",
                         color_discrete_sequence=[_ACCENT2],
                         title=f"Publications per year — {selected_concept}")
            fig.update_layout(**_layout(height=260, xaxis=dict(tickmode="linear", dtick=1)))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — Domain Map
# ============================================================
with tab_domain:
    st.markdown("#### Domain Map — What Each Concept Talks About")
    st.caption(
        "Computed from TF-IDF keyword weights. "
        "Similar concepts appear close together; colour encodes concept family."
    )

    # â”€â”€ Build concept Ã— keyword TF-IDF matrix from precomputed keywords â”€â”€â”€â”€â”€â”€
    def _build_concept_matrix(kw_df: pd.DataFrame):
        """
        Pivot keywords_by_concept into a concept Ã— keyword matrix.
        Returns (concept_names, numpy matrix) or ([], None) if data missing.
        """
        if kw_df.empty or not {"concept_name", "keyword", "weight"}.issubset(kw_df.columns):
            return [], None
        pivot = (kw_df.pivot_table(index="concept_name", columns="keyword",
                                   values="weight", aggfunc="mean")
                 .fillna(0))
        concepts = pivot.index.tolist()
        mat = normalize(pivot.values, norm="l2")
        return concepts, mat

    concepts_list, kw_matrix = _build_concept_matrix(keywords_df)

    # â”€â”€ 3a. Concept Similarity Heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if kw_matrix is not None and len(concepts_list) > 1:
        sim = cosine_similarity(kw_matrix)
        sim_df = pd.DataFrame(sim, index=concepts_list, columns=concepts_list)

        st.markdown("##### Concept Similarity Heatmap")
        st.caption(
            "Cosine similarity of TF-IDF keyword profiles. "
            "1.0 = identical vocabulary, 0.0 = no overlap."
        )
        fig_sim = px.imshow(
            sim_df,
            color_continuous_scale="RdBu_r",
            range_color=[0, 1],
            labels=dict(color="Similarity"),
            title="Keyword similarity between concepts",
            aspect="auto",
            text_auto=".2f",
        )
        fig_sim.update_layout(**_layout(height=460, margin=dict(l=10, r=10, t=40, b=10)))
        fig_sim.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        fig_sim.update_yaxes(tickfont=dict(size=10))
        fig_sim.update_traces(textfont=dict(size=9))
        st.plotly_chart(fig_sim, use_container_width=True)

    # â”€â”€ 3b. PCA Cluster Scatter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if kw_matrix is not None and len(concepts_list) >= 3:
        st.markdown("##### Concept Cluster Map (PCA)")
        st.caption(
            "Each point is a concept projected onto its two principal keyword axes. "
            "Concepts near each other share vocabulary."
        )
        n_comp = min(2, kw_matrix.shape[1], kw_matrix.shape[0])
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(kw_matrix)
        var = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "Concept": concepts_list,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1] if n_comp > 1 else [0.0] * len(concepts_list),
            "highlight": ["â–¶ " + c if c == selected_concept else c for c in concepts_list],
        })

        # Add chunk counts as marker size
        if not concept_counts_df.empty:
            pca_df = pca_df.merge(
                concept_counts_df[["concept_name", "chunk_count"]].rename(
                    columns={"concept_name": "Concept"}
                ),
                on="Concept", how="left",
            )
            pca_df["chunk_count"] = pca_df["chunk_count"].fillna(1)
        else:
            pca_df["chunk_count"] = 10

        fig_pca = px.scatter(
            pca_df,
            x="PC1", y="PC2",
            size="chunk_count",
            color="Concept",
            size_max=46,
            labels={
                "PC1": f"PC 1 ({var[0]*100:.0f}% var)",
                "PC2": f"PC 2 ({var[1]*100:.0f}% var)" if n_comp > 1 else "PC 2",
            },
            title="Concept cluster map — keyword space",
        )
        fig_pca.update_traces(
            marker=dict(opacity=0.75, line=dict(width=1.5, color="#FFFFFF")),
            showlegend=False,
        )
        # Add per-point annotations offset away from the bubble centre
        # so labels never sit inside a bubble
        for _, r in pca_df.iterrows():
            is_sel = r["Concept"] == selected_concept
            fig_pca.add_annotation(
                x=r["PC1"],
                y=r["PC2"],
                text=f"<b>{r['Concept']}</b>" if is_sel else r["Concept"],
                showarrow=True,
                arrowhead=0,
                arrowwidth=0.8,
                arrowcolor="#9CA3AF",
                ax=0,
                ay=-36,
                font=dict(family=_FONT, size=10 if not is_sel else 11,
                          color=_ACCENT if is_sel else _TEXT),
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=2,
            )
        fig_pca.update_layout(**_layout(
            height=520,
            xaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=True,
                       zerolinecolor=_BORDER, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor=_BORDER, zeroline=True,
                       zerolinecolor=_BORDER, tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig_pca, use_container_width=True)

    # â”€â”€ 3c. Multi-line concept trend over years â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if has_year and not concepts_df.empty and not documents_df.empty:
        st.markdown("##### Concept Activity Over Time (Multi-line)")
        st.caption(
            "Chunk count per concept per year. "
            "Shows which topics gained or lost research attention over time."
        )
        yc_trend = (
            concepts_df[["doc_id", "concept_name", "chunk_id"]].drop_duplicates(
                subset=["doc_id", "concept_name", "chunk_id"]
            )
            .merge(
                documents_df[["doc_id", "year"]].dropna(subset=["year"]),
                on="doc_id", how="inner",
            )
        )
        if not yc_trend.empty:
            yc_trend["year"] = yc_trend["year"].astype(int)
            trend_agg = (
                yc_trend.groupby(["year", "concept_name"])["chunk_id"]
                .count()
                .reset_index(name="chunks")
                .sort_values("year")
            )
            # highlight selected concept with a thicker line
            trend_agg["_lw"] = trend_agg["concept_name"].apply(
                lambda c: 3 if c == selected_concept else 1.2
            )
            fig_trend = px.line(
                trend_agg,
                x="year", y="chunks",
                color="concept_name",
                line_group="concept_name",
                markers=True,
                labels={"chunks": "Chunks", "year": "Year", "concept_name": "Concept"},
                title="Research activity trend by concept",
            )
            # Bold the selected concept
            for trace in fig_trend.data:
                if trace.name == selected_concept:
                    trace.line = dict(width=3)
                    trace.marker = dict(size=7)
                else:
                    trace.line = dict(width=1.5)
                    trace.marker = dict(size=4)
            fig_trend.update_layout(**_layout(
                height=420,
                xaxis=dict(tickmode="linear", dtick=1),
                legend=dict(
                    title=None,          # removes the "Concept" legend title
                    orientation="h",
                    yanchor="top",
                    y=-0.18,             # push legend below the x-axis
                    xanchor="left",
                    x=0,
                    font=dict(size=10),
                ),
                margin=dict(l=10, r=10, t=44, b=10),
            ))
            st.plotly_chart(fig_trend, use_container_width=True)

    # â”€â”€ 3d. Per-concept keyword fingerprint (small multiples) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not keywords_df.empty:
        st.markdown("##### Keyword Fingerprint per Concept")
        st.caption(
            "Top 8 TF-IDF keywords for each concept shown as individual panels. "
            "Each bar is that keyword's weight within that concept's corpus."
        )
        # Top-8 keywords per concept, ordered by weight
        fp_df = (
            keywords_df.sort_values(["concept_name", "weight"], ascending=[True, False])
            .groupby("concept_name")
            .head(8)
            .copy()
        )
        if not fp_df.empty:
            # colour the selected concept's panel bars differently
            fp_df["bar_color"] = fp_df["concept_name"].apply(
                lambda c: _ACCENT if c == selected_concept else _ACCENT2
            )
            n_concepts = fp_df["concept_name"].nunique()
            # two columns of subplots via facet_col
            fig_fp = px.bar(
                fp_df.sort_values("weight", ascending=True),
                x="weight",
                y="keyword",
                facet_col="concept_name",
                facet_col_wrap=2,
                orientation="h",
                color="concept_name",
                labels={"weight": "", "keyword": "", "concept_name": ""},
                title="Keyword fingerprint per concept",
                height=max(320, n_concepts * 150),
            )
            # Clean up facet titles — strip the "concept_name=" prefix
            fig_fp.for_each_annotation(
                lambda a: a.update(
                    text=a.text.split("=")[-1],
                    font=dict(size=11, family=_FONT, color=_TEXT),
                )
            )
            fig_fp.update_traces(showlegend=False)
            fig_fp.update_xaxes(showgrid=True, gridcolor=_BORDER,
                                zeroline=False, tickfont=dict(size=9))
            fig_fp.update_yaxes(tickfont=dict(size=9), matches=None)
            fig_fp.update_layout(
                font=dict(family=_FONT, size=11, color=_TEXT),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=48, b=10),
                title_font=dict(family=_FONT, size=13, color=_MUTED),
                title_x=0,
                showlegend=False,
            )
            st.plotly_chart(fig_fp, use_container_width=True)

# ============================================================
# TAB 4 — Cross-Concept Comparisons
# ============================================================
with tab_compare:
    cc1, cc2 = st.columns(2)

    with cc1:
        if not concepts_df.empty and "confidence_score" in concepts_df.columns:
            avg_c = (concepts_df.groupby("concept_name")["confidence_score"]
                     .mean().reset_index(name="avg_confidence")
                     .sort_values("avg_confidence", ascending=True))
            avg_c["_s"] = avg_c["concept_name"].apply(lambda c: "Selected" if c == selected_concept else "Other")
            fig = px.bar(avg_c, x="avg_confidence", y="concept_name", orientation="h",
                         color="_s", color_discrete_map={"Selected": _ACCENT, "Other": "#D1D5DB"},
                         labels={"avg_confidence": "Avg confidence", "concept_name": ""},
                         title="Average classification confidence")
            fig.update_traces(showlegend=False)
            fig.update_layout(**_layout(height=370))
            st.plotly_chart(fig, use_container_width=True)

    with cc2:
        if (not concepts_df.empty and not documents_df.empty
                and "page_count" in documents_df.columns):
            dcu = (concepts_df[["concept_name", "doc_id"]].drop_duplicates()
                   .merge(documents_df[["doc_id", "page_count"]], on="doc_id", how="left"))
            ap = (dcu.groupby("concept_name")["page_count"]
                  .mean().reset_index(name="avg_pages")
                  .sort_values("avg_pages", ascending=True))
            fig = px.bar(ap, x="avg_pages", y="concept_name", orientation="h",
                         color="avg_pages",
                         color_continuous_scale=[[0, "#FEF3C7"], [1, "#B45309"]],
                         labels={"avg_pages": "Avg pages", "concept_name": ""},
                         title="Average document length (pages)")
            fig.update_layout(**_layout(height=370, coloraxis_showscale=False))
            st.plotly_chart(fig, use_container_width=True)

    if not concepts_df.empty and not documents_df.empty:
        st.markdown("#### Concept Composition per Document")
        st.caption("Top 15 most-covered documents — coloured by concept contribution.")
        dcc = (concepts_df.groupby(["doc_id", "concept_name"])["chunk_id"]
               .nunique().reset_index(name="chunks"))
        top15 = dcc.groupby("doc_id")["chunks"].sum().nlargest(15).index.tolist()
        dct = dcc[dcc["doc_id"].isin(top15)].copy()
        ls = documents_df[["doc_id", "title", "filename"]].copy()
        ls["label"] = ls["title"].fillna(ls["filename"]).astype(str).str[:45]
        dct = dct.merge(ls[["doc_id", "label"]], on="doc_id", how="left")
        fig = px.bar(dct, x="chunks", y="label", color="concept_name",
                     orientation="h", barmode="stack",
                     labels={"chunks": "Chunks", "label": "", "concept_name": "Concept"},
                     title="Concept mix per document")
        fig.update_layout(**_layout(
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=10)),
        ))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 — Documents
# ============================================================
with tab_docs:
    st.markdown(f"#### Related Documents — {selected_concept}")
    st.caption(f"{len(concept_doc_ids)} document(s) matched")

    if not concept_doc_ids:
        st.info("No documents matched this concept with the current filters.")
    elif not documents_df.empty:
        rel = documents_df[documents_df["doc_id"].isin(concept_doc_ids)].copy()
        if not concept_rows.empty:
            dcc2 = (concept_rows.groupby("doc_id")["chunk_id"]
                    .nunique().reset_index(name="matching_chunks"))
            rel = rel.merge(dcc2, on="doc_id", how="left").sort_values(
                "matching_chunks", ascending=False)
        dcols = ["filename", "title", "authors", "year", "page_count", "matching_chunks"]
        dcols = [c for c in dcols if c in rel.columns]
        st.dataframe(rel[dcols].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

# ============================================================
# TAB 6 — Source Snippets  (doc-level selection + keyword filter)
# ============================================================
with tab_snippets:
    st.markdown(f"#### Source Snippets — {selected_concept}")

    if chunks_df.empty or not concept_chunk_ids:
        st.info("No snippets available for this concept.")
    else:
        # Build a doc â†’ filename lookup for the selectbox
        doc_options: dict[str, list[int]] = {}   # label â†’ list of chunk_ids
        doc_options["All documents"] = list(concept_chunk_ids)

        if not concept_rows.empty and not documents_df.empty:
            # per-doc chunk counts for the selector label
            per_doc = (concept_rows.groupby("doc_id")["chunk_id"]
                       .nunique().reset_index(name="n"))
            per_doc = per_doc.merge(
                documents_df[["doc_id", "filename"]].drop_duplicates(),
                on="doc_id", how="left"
            ).sort_values("n", ascending=False)
            for _, r in per_doc.iterrows():
                label = f"{r['filename']}  ({int(r['n'])} chunks)"
                doc_chunk_ids = set(
                    concept_rows.loc[concept_rows["doc_id"] == r["doc_id"], "chunk_id"]
                )
                doc_options[label] = list(doc_chunk_ids)

        sc1, sc2 = st.columns([3, 1])
        with sc1:
            doc_sel = st.selectbox("Filter by document", options=list(doc_options.keys()))
        with sc2:
            conf_threshold = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

        active_ids: set[int] = set(doc_options[doc_sel])

        # Pull and filter snippets
        snip = chunks_df[chunks_df["chunk_id"].isin(active_ids)].copy()

        if not concept_rows.empty:
            cf = concept_rows[["chunk_id", "confidence_score"]].drop_duplicates("chunk_id")
            snip = snip.merge(cf, on="chunk_id", how="left")
            if conf_threshold > 0:
                snip = snip[snip["confidence_score"].fillna(0) >= conf_threshold]
            snip = snip.sort_values("confidence_score", ascending=False)

        if keyword_search.strip():
            snip = snip[snip["chunk_text"].str.contains(
                re.escape(keyword_search.strip()), case=False, na=False)]

        total = len(snip)
        limit = 30
        st.caption(
            f"{total:,} snippet(s) matched"
            + (f" — showing top {limit}" if total > limit else "")
        )

        if snip.empty:
            st.info("No snippets match the current filters.")
        else:
            for _, row in snip.head(limit).iterrows():
                cv = row.get("confidence_score")
                badge = (
                    f'&nbsp;<span style="background:#EFF6FF;color:{_ACCENT};'
                    f'font-size:.67rem;padding:1px 5px;border-radius:999px;font-weight:600">'
                    f'{cv:.3f}</span>'
                    if cv is not None and pd.notna(cv) else ""
                )
                text = str(row["chunk_text"])
                if keyword_search.strip():
                    text = re.sub(
                        f"({re.escape(keyword_search.strip())})",
                        r"<mark>\1</mark>", text, flags=re.IGNORECASE,
                    )
                st.markdown(
                    f'<div class="co2m-snippet">'
                    f'<div class="meta">'
                    f'{row["filename"]}&nbsp;Â·&nbsp;p.{row["page_number"]}{badge}'
                    f'</div>'
                    f'<div class="body">{text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ============================================================
# TAB 8 — System Verification
# ============================================================
with tab_verify:
    st.markdown("### System Verification")
    st.caption("Check the live Postgres RAG backend and optional verification module")

    if RAGQueryService is None:
        st.warning("Verification is unavailable because semantic search dependencies could not be loaded.")
        if _RAG_IMPORT_ERROR is not None:
            st.code(str(_RAG_IMPORT_ERROR))
    else:
        try:
            service = get_query_service()
            stats = service.get_stats()
            llm_status = service.get_llm_status()
            v1, v2, v3 = st.columns(3)
            v1.metric("Postgres Documents", f"{stats['total_documents']:,}")
            v2.metric("Embedded Chunks", f"{stats['embedded_chunks']:,}")
            v3.metric("Unique Documents", f"{stats['unique_documents']:,}")
            st.caption(
                f"Backend: {stats.get('backend')} Â· "
                f"LLM: {llm_status.get('provider')} "
                f"({'available' if llm_status.get('available') else 'unavailable'})"
            )
            if RAGVerification is None:
                st.info("Detailed verification is unavailable because the optional verification module is not installed.")
                if _VERIFICATION_IMPORT_ERROR is not None:
                    st.code(str(_VERIFICATION_IMPORT_ERROR))
            else:
                verifier = RAGVerification(query_service=service)
                if st.button("Run Verification", key="verify_btn", use_container_width=True):
                    st.session_state.verification_results = verifier.run_all_verifications()
                if "verification_results" in st.session_state:
                    st.json(st.session_state.verification_results)
        except Exception as exc:
            st.error(f"Verification failed: {exc}")
