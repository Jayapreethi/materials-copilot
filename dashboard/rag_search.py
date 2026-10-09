"""
dashboard/rag_search.py
Streamlit RAG search interface for CO2M corpus.
Integrates semantic search with vector database.
"""

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from rag_pipeline.query_service import RAGQueryService

logger = logging.getLogger(__name__)


def display_rag_search():
    """Display RAG search interface in Streamlit."""

    st.markdown("### Semantic Search")

    # Initialize service (cached)
    @st.cache_resource
    def get_query_service():
        """Load query service once."""
        try:
            service = RAGQueryService(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                vector_store_path="./vector_db/data",
                collection_name="co2m_corpus",
                device="cpu",
            )
            logger.info("RAG Query Service initialized")
            return service
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise

    try:
        query_service = get_query_service()
    except Exception as e:
        st.error(f"Failed to initialize semantic search system: {e}")
        st.info("Make sure the vector database is populated. Run: `python ingest_enhanced.py`")
        return

    # Get stats
    try:
        stats = query_service.get_stats()
        st.info(
            f"Vector DB: {stats.get('total_documents', 0):,} documents indexed"
        )
    except Exception as e:
        logger.warning(f"Could not retrieve stats: {e}")

    # Search input
    query = st.text_input(
        "Enter your question or search term:",
        placeholder="e.g., 'CO2 capture materials', 'amine-based sorbents', 'separation efficiency'",
        key="rag_query",
    )

    # Search parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5, key="top_k_rag")
    with col2:
        include_context = st.checkbox(
            "Generate context summary",
            value=True,
            help="Format results as context for LLM processing",
            key="include_context_rag",
        )

    # Optional concept filter
    concept_filter = st.selectbox(
        "Filter by concept (optional):",
        options=["None"],  # Will be replaced if concepts are available
        key="concept_filter_rag",
    )

    if concept_filter == "None":
        concept_filter = None

    # Execute search
    if query and st.button("Search", key="search_button"):
        try:
            with st.spinner("Searching..."):
                results = query_service.query(
                    question=query,
                    top_k=top_k,
                    concept_filter=concept_filter,
                    include_context=include_context,
                )

            # Display results
            st.success(f"Found {results['num_results']} relevant documents")

            # Display summary if context was generated
            if "context_summary" in results:
                summary = results["context_summary"]
                with st.expander("Summary Statistics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Results", summary["total_results"])
                    with col2:
                        st.metric("Unique Docs", summary["unique_documents"])
                    with col3:
                        st.metric("Avg Similarity", f"{summary['avg_similarity']:.3f}")

                    if summary["materials_found"] > 0:
                        st.metric("Materials Found", summary["materials_found"])

                    if "co2_uptake_stats" in summary:
                        co2_stats = summary["co2_uptake_stats"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CO2 Avg (mmol/g)", f"{co2_stats['avg']:.2f}")
                        with col2:
                            st.metric("CO2 Min", f"{co2_stats['min']:.2f}")
                        with col3:
                            st.metric("CO2 Max", f"{co2_stats['max']:.2f}")

            # Display context
            if "context" in results:
                with st.expander("Formatted Context", expanded=False):
                    st.text_area(
                        "Context",
                        value=results["context"],
                        height=300,
                        disabled=True,
                    )

            # Display individual results
            st.markdown("#### Results")
            for result in results["results"]:
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        similarity_pct = result["similarity"] * 100
                        st.metric(
                            f"#{result['rank']}",
                            f"{similarity_pct:.1f}%",
                            delta="relevance",
                        )
                    with col2:
                        st.markdown(
                            f"**{result['filename']}** (Page {result['page_number']})\n\n"
                            f"{result['text'][:300]}...\n\n"
                        )
                        if result["material_name"]:
                            st.caption(f"Material: {result['material_name']}")
                        if result.get("co2_uptake_mmol_g"):
                            st.caption(f"CO₂ Uptake: {result['co2_uptake_mmol_g']} mmol/g")

                st.divider()

        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.exception("Search error")

    # Example queries
    with st.expander("Example Queries", expanded=False):
        st.markdown(
            """
            - What materials are used for CO2 capture?
            - How does amine-based separation work?
            - What are the latest advances in sorbent materials?
            - Compare MOF and zeolite performance
            """
        )
