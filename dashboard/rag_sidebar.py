"""
dashboard/rag_sidebar.py
RAG search sidebar widget for CO2M dashboard.
"""

import logging

import streamlit as st

from rag_pipeline.query_service import RAGQueryService

logger = logging.getLogger(__name__)


def display_rag_sidebar():
    """Display RAG search in sidebar."""

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Semantic Search")

    # Query input
    query = st.sidebar.text_input(
        "Search corpus:",
        placeholder="natural language query",
        key="rag_sidebar_query",
    )

    # Quick search button
    if query and st.sidebar.button("Search", key="rag_sidebar_search"):
        try:
            @st.cache_resource
            def get_service():
                return RAGQueryService()

            service = get_service()

            # Perform search
            results = service.query(query, top_k=3)

            # Store in session state for main area
            st.session_state.rag_results = results
            st.session_state.show_rag_results = True

        except Exception as e:
            logger.error(f"RAG sidebar search error: {e}")
            st.sidebar.error(f"Search failed: {e}")

    # Display results if available
    if st.session_state.get("show_rag_results") and "rag_results" in st.session_state:
        results = st.session_state.rag_results
        st.sidebar.markdown(f"**Found {results['num_results']} results**")

        for i, result in enumerate(results["results"][:3], 1):
            with st.sidebar.expander(f"[{i}] {result['filename']} ({result['similarity']:.2%})"):
                st.caption(result["text"][:200] + "...")
                if result.get("material_name"):
                    st.caption(f"Material: {result['material_name']}")
