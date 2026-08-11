#!/usr/bin/env python
"""
test_rag_pipeline.py
Tests for the RAG pipeline components.
"""

import logging
import unittest
from pathlib import Path

from rag_pipeline.retriever import RAGRetriever
from rag_pipeline.query_service import RAGQueryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRAGRetriever(unittest.TestCase):
    """Test RAG Retriever component."""

    @classmethod
    def setUpClass(cls):
        """Initialize retriever for tests."""
        cls.retriever = RAGRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_path="./vector_db/data",
            collection_name="co2m_corpus",
            device="cpu",
        )

    def test_retriever_initialization(self):
        """Test retriever initializes correctly."""
        self.assertIsNotNone(self.retriever)
        stats = self.retriever.get_stats()
        self.assertIn("total_documents", stats)
        self.assertGreater(stats["total_documents"], 0)

    def test_single_query_retrieval(self):
        """Test retrieving results for a single query."""
        query = "CO2 capture materials"
        results = self.retriever.retrieve(query, top_k=5)

        self.assertIn("results", results)
        self.assertIn("query", results)
        self.assertEqual(results["query"], query)
        self.assertLessEqual(len(results["results"]), 5)

        if results["results"]:
            result = results["results"][0]
            self.assertIn("rank", result)
            self.assertIn("similarity", result)
            self.assertIn("text", result)
            self.assertIn("filename", result)

    def test_retrieval_similarity_range(self):
        """Test that similarity scores are in valid range [0, 1]."""
        results = self.retriever.retrieve("material properties", top_k=5)

        for result in results["results"]:
            self.assertGreaterEqual(result["similarity"], 0.0)
            self.assertLessEqual(result["similarity"], 1.0)

    def test_batch_retrieval(self):
        """Test batch retrieval for multiple queries."""
        queries = ["CO2 capture", "sorbent materials", "separation efficiency"]
        results = self.retriever.batch_retrieve(queries, top_k=3)

        self.assertEqual(len(results), len(queries))
        for result in results:
            self.assertIn("results", result)


class TestRAGQueryService(unittest.TestCase):
    """Test RAG Query Service component."""

    @classmethod
    def setUpClass(cls):
        """Initialize service for tests."""
        cls.service = RAGQueryService(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vector_store_path="./vector_db/data",
            collection_name="co2m_corpus",
            device="cpu",
        )

    def test_service_initialization(self):
        """Test service initializes correctly."""
        self.assertIsNotNone(self.service)
        stats = self.service.get_stats()
        self.assertIsNotNone(stats)

    def test_query_with_context(self):
        """Test querying with context generation."""
        query = "What materials are used for CO2 capture?"
        results = self.service.query(
            question=query,
            top_k=5,
            include_context=True,
        )

        self.assertEqual(results["question"], query)
        self.assertIn("context", results)
        self.assertIn("context_summary", results)
        self.assertGreater(len(results["context"]), 0)

    def test_query_context_summary(self):
        """Test context summary generation."""
        results = self.service.query(
            question="CO2 separation",
            top_k=5,
            include_context=True,
        )

        summary = results["context_summary"]
        self.assertIn("total_results", summary)
        self.assertIn("unique_documents", summary)
        self.assertIn("avg_similarity", summary)
        self.assertGreaterEqual(summary["total_results"], 0)
        self.assertGreaterEqual(summary["unique_documents"], 0)

    def test_topic_search(self):
        """Test topic search functionality."""
        results = self.service.search_by_topic("amine sorbents", top_k=5)

        self.assertIn("topic", results)
        self.assertIn("results", results)
        self.assertIn("count", results)

    def test_get_related_documents(self):
        """Test finding related documents."""
        sample_text = "This is a document about CO2 capture using advanced materials."
        results = self.service.get_related_documents(sample_text, top_k=5)

        self.assertIn("related_documents", results)
        self.assertIn("count", results)
        self.assertIn("query_length", results)


if __name__ == "__main__":
    unittest.main()
