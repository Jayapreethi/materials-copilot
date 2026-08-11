"""
RAG Retriever - Semantic search and retrieval from vector database.
"""

import logging
from typing import Any, Dict, List, Optional

from embeddings.embeddings import EmbeddingGenerator
from vector_db.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Semantic retriever using embeddings and vector store."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "./vector_db/data",
        collection_name: str = "co2m_corpus",
        device: str = "cpu",
    ):
        """
        Initialize RAG retriever.

        Args:
            embedding_model: HuggingFace model ID for embeddings
            vector_store_path: Path to Chroma vector store
            collection_name: Name of the collection
            device: "cpu" or "cuda" for embedding computation
        """
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            device=device,
        )
        self.vector_store = VectorStore(
            persist_directory=vector_store_path,
            collection_name=collection_name,
        )
        logger.info(f"RAGRetriever initialized with {self.vector_store.get_stats()}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        concept_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k relevant documents for a query.

        Args:
            query: Natural language query
            top_k: Number of results to return
            concept_filter: Optional concept name to filter by

        Returns:
            Dict with 'results' (list of result dicts) and 'query'
        """
        logger.debug(f"Retrieving for query: {query} (top_k={top_k})")

        # Embed the query
        query_embedding = self.embedding_generator.embed_single(query).tolist()

        # Build where filter if concept specified
        where = None
        if concept_filter:
            where = {"concept_name": {"$eq": concept_filter}}
            logger.debug(f"Filtering by concept: {concept_filter}")

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        # Format results
        formatted_results = []
        for idx, (doc_id, distance, text, metadata) in enumerate(
            zip(
                results["ids"],
                results["distances"],
                results["documents"],
                results["metadatas"],
            )
        ):
            similarity = 1 - distance  # Convert distance to similarity
            formatted_results.append({
                "rank": idx + 1,
                "doc_id": doc_id,
                "similarity": round(similarity, 4),
                "text": text,
                "filename": metadata.get("filename", "unknown"),
                "page_number": metadata.get("page_number", None),
                "material_name": metadata.get("material_name"),
                "co2_uptake_mmol_g": metadata.get("co2_uptake_mmol_g"),
                "metadata": metadata,
            })

        return {
            "query": query,
            "concept_filter": concept_filter,
            "results": formatted_results,
            "total_results": len(formatted_results),
        }

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        concept_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query
            concept_filter: Optional concept filter

        Returns:
            List of result dicts (same format as retrieve)
        """
        logger.info(f"Batch retrieving for {len(queries)} queries")
        return [self.retrieve(q, top_k, concept_filter) for q in queries]

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        store_stats = self.vector_store.get_stats()
        return {
            **store_stats,
            "embedding_model": self.embedding_generator.model_name,
            "embedding_dimension": self.embedding_generator.dimension,
        }
