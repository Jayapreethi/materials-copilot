"""
verification.py
Verification and evaluation methods for the RAG pipeline.
"""

import logging
import time
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGVerification:
    """Verification and evaluation suite for RAG pipeline."""

    def __init__(self, query_service=None):
        """Initialize verification suite.
        
        Args:
            query_service: RAGQueryService instance to verify.
        """
        self.service = query_service
        self.results = {}

    def verify_vector_db(self) -> Dict[str, Any]:
        """Verify vector database status and connectivity.
        
        Returns:
            Dict with status, document count, collection info.
        """
        result = {
            "status": "unknown",
            "document_count": 0,
            "collection_name": "unknown",
            "embedding_model": "unknown",
            "error": None,
        }

        try:
            if not self.service:
                result["error"] = "Query service not initialized"
                result["status"] = "failed"
                return result

            stats = self.service.get_stats()
            result["document_count"] = stats.get("total_documents", 0)
            result["collection_name"] = stats.get("collection_name", "unknown")
            result["embedding_model"] = stats.get("embedding_model", "unknown")

            if result["document_count"] > 0:
                result["status"] = "healthy"
            else:
                result["status"] = "empty"
                result["error"] = "No documents indexed"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Vector DB verification failed: {e}")

        return result

    def test_sample_query(self, query: str = "CO2 capture materials", top_k: int = 5) -> Dict[str, Any]:
        """Test RAG pipeline with a sample query.
        
        Args:
            query: Test query string.
            top_k: Number of results to retrieve.
            
        Returns:
            Dict with query results, latency, and quality metrics.
        """
        result = {
            "query": query,
            "status": "unknown",
            "results_count": 0,
            "latency_ms": 0,
            "avg_similarity": 0.0,
            "top_result_file": None,
            "top_result_similarity": 0.0,
            "error": None,
        }

        try:
            if not self.service:
                result["error"] = "Query service not initialized"
                result["status"] = "failed"
                return result

            # Measure query latency
            start_time = time.time()
            query_result = self.service.query(
                question=query,
                top_k=top_k,
                include_context=False,
            )
            elapsed = time.time() - start_time
            result["latency_ms"] = round(elapsed * 1000, 2)

            # Extract results
            results = query_result.get("results", [])
            result["results_count"] = len(results)

            if results:
                similarities = [r.get("similarity", 0) for r in results]
                result["avg_similarity"] = round(sum(similarities) / len(similarities), 3)
                result["top_result_similarity"] = round(results[0].get("similarity", 0), 3)
                result["top_result_file"] = results[0].get("filename", "unknown")
                result["status"] = "success"
            else:
                result["status"] = "no_results"
                result["error"] = "Query returned no results"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Sample query test failed: {e}")

        return result

    def test_batch_queries(self, queries: List[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """Test RAG pipeline with multiple queries.
        
        Args:
            queries: List of test queries.
            top_k: Number of results per query.
            
        Returns:
            Dict with batch test results.
        """
        if not queries:
            queries = [
                "CO2 capture materials",
                "amine-based sorbents",
                "metal-organic frameworks",
            ]

        result = {
            "status": "unknown",
            "queries_count": len(queries),
            "successful": 0,
            "failed": 0,
            "avg_latency_ms": 0.0,
            "avg_similarity": 0.0,
            "errors": [],
        }

        try:
            if not self.service:
                result["error"] = "Query service not initialized"
                result["status"] = "failed"
                return result

            latencies = []
            similarities = []

            for query in queries:
                try:
                    start_time = time.time()
                    query_result = self.service.query(
                        question=query,
                        top_k=top_k,
                        include_context=False,
                    )
                    elapsed = time.time() - start_time
                    latencies.append(elapsed * 1000)

                    results = query_result.get("results", [])
                    if results:
                        result["successful"] += 1
                        sims = [r.get("similarity", 0) for r in results]
                        similarities.extend(sims)
                    else:
                        result["failed"] += 1
                        result["errors"].append(f"{query}: no results")

                except Exception as e:
                    result["failed"] += 1
                    result["errors"].append(f"{query}: {str(e)}")

            if result["successful"] > 0:
                result["status"] = "success" if result["failed"] == 0 else "partial"
                result["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
                result["avg_similarity"] = round(sum(similarities) / len(similarities), 3)
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "failed"
            result["errors"] = [str(e)]
            logger.error(f"Batch query test failed: {e}")

        return result

    def test_retrieval_quality(self, num_samples: int = 5) -> Dict[str, Any]:
        """Test retrieval quality across different queries.
        
        Args:
            num_samples: Number of sample queries to test.
            
        Returns:
            Dict with quality metrics.
        """
        test_queries = [
            "CO2 capture materials",
            "amine-based sorbents",
            "metal-organic frameworks",
            "separation efficiency",
            "sorbent performance",
        ][:num_samples]

        result = {
            "status": "unknown",
            "samples_tested": 0,
            "avg_top_result_similarity": 0.0,
            "min_similarity": 1.0,
            "max_similarity": 0.0,
            "median_similarity": 0.0,
            "relevance_score": 0.0,
            "error": None,
        }

        try:
            all_similarities = []

            for query in test_queries:
                query_result = self.test_sample_query(query, top_k=5)
                if query_result["status"] == "success":
                    result["samples_tested"] += 1
                    sim = query_result["top_result_similarity"]
                    all_similarities.append(sim)

            if all_similarities:
                result["status"] = "success"
                result["avg_top_result_similarity"] = round(sum(all_similarities) / len(all_similarities), 3)
                result["min_similarity"] = round(min(all_similarities), 3)
                result["max_similarity"] = round(max(all_similarities), 3)
                
                # Calculate median
                sorted_sims = sorted(all_similarities)
                mid = len(sorted_sims) // 2
                result["median_similarity"] = round(
                    (sorted_sims[mid] + sorted_sims[mid - 1]) / 2 if len(sorted_sims) % 2 == 0 else sorted_sims[mid],
                    3
                )
                
                # Relevance score: 0.7+ is good quality
                result["relevance_score"] = round(result["avg_top_result_similarity"], 3)
            else:
                result["status"] = "failed"
                result["error"] = "No successful queries"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Retrieval quality test failed: {e}")

        return result

    def calculate_reliability_score(self) -> Dict[str, Any]:
        """Calculate overall reliability score based on verification results.
        
        Returns:
            Dict with reliability score (0-100) and component breakdown.
        """
        if not self.results:
            return {
                "score": 0,
                "percentage": "0%",
                "status": "not_run",
                "components": {},
                "breakdown": "No verification results available"
            }

        # Extract results
        db = self.results.get("vector_db", {})
        sample = self.results.get("sample_query", {})
        batch = self.results.get("batch_queries", {})
        quality = self.results.get("retrieval_quality", {})

        # Calculate component scores (0-1 range)
        components = {}

        # 1. Database Health (25% weight)
        db_health = 0.0
        if db.get("status") == "healthy":
            db_health = 1.0
        elif db.get("status") == "degraded":
            db_health = 0.5
        elif db.get("document_count", 0) > 0:
            db_health = 0.7
        components["database_health"] = db_health

        # 2. Query Success Rate (25% weight)
        query_success = 0.0
        if batch.get("queries_count", 0) > 0:
            query_success = batch.get("successful", 0) / batch.get("queries_count", 1)
        components["query_success"] = query_success

        # 3. Retrieval Quality (30% weight)
        quality_score = 0.0
        relevance = quality.get("relevance_score", 0.0)
        # Quality score: normalize relevance (0.3+ is acceptable, 0.7+ is excellent)
        if relevance >= 0.7:
            quality_score = 1.0
        elif relevance >= 0.5:
            quality_score = 0.8
        elif relevance >= 0.3:
            quality_score = 0.5
        else:
            quality_score = max(0.2, relevance)
        components["retrieval_quality"] = quality_score

        # 4. Performance Efficiency (20% weight)
        perf_score = 1.0
        latency = sample.get("latency_ms", 0)
        # Excellent: <50ms, Good: <200ms, Acceptable: <500ms
        if latency > 1000:
            perf_score = 0.4
        elif latency > 500:
            perf_score = 0.6
        elif latency > 200:
            perf_score = 0.8
        elif latency > 50:
            perf_score = 0.95
        components["performance"] = perf_score

        # Calculate weighted score
        weights = {
            "database_health": 0.25,
            "query_success": 0.25,
            "retrieval_quality": 0.30,
            "performance": 0.20,
        }

        overall_score = sum(components.get(key, 0) * weight for key, weight in weights.items())
        overall_score = round(overall_score * 100, 1)  # Convert to 0-100 scale

        # Determine status
        if overall_score >= 85:
            status = "Excellent"
            color_code = "#10B981"  # Green
        elif overall_score >= 70:
            status = "Good"
            color_code = "#3B82F6"  # Blue
        elif overall_score >= 50:
            status = "Fair"
            color_code = "#F59E0B"  # Amber
        else:
            status = "Poor"
            color_code = "#EF4444"  # Red

        return {
            "score": overall_score,
            "percentage": f"{overall_score}%",
            "status": status,
            "color": color_code,
            "components": {
                "database_health": round(components["database_health"] * 100, 1),
                "query_success": round(components["query_success"] * 100, 1),
                "retrieval_quality": round(components["retrieval_quality"] * 100, 1),
                "performance": round(components["performance"] * 100, 1),
            },
            "breakdown": (
                f"Database Health: {components['database_health']*100:.0f}% | "
                f"Query Success: {components['query_success']*100:.0f}% | "
                f"Retrieval Quality: {components['retrieval_quality']*100:.0f}% | "
                f"Performance: {components['performance']*100:.0f}%"
            ),
        }

    def run_all_verifications(self) -> Dict[str, Dict[str, Any]]:
        """Run all verification tests.
        
        Returns:
            Dict containing results of all verification tests.
        """
        logger.info("Starting comprehensive RAG verification...")

        self.results = {
            "vector_db": self.verify_vector_db(),
            "sample_query": self.test_sample_query(),
            "batch_queries": self.test_batch_queries(),
            "retrieval_quality": self.test_retrieval_quality(),
        }

        # Overall status
        all_healthy = all(
            v.get("status") == "success" or v.get("status") == "healthy"
            for v in self.results.values()
        )
        self.results["overall_status"] = "healthy" if all_healthy else "degraded"

        # Calculate reliability score
        self.results["reliability_score"] = self.calculate_reliability_score()

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of verification results.
        
        Returns:
            Dict with key metrics and health status.
        """
        if not self.results:
            return {"status": "not_run", "message": "Run verifications first"}

        db_status = self.results.get("vector_db", {})
        quality = self.results.get("retrieval_quality", {})
        batch = self.results.get("batch_queries", {})

        summary = {
            "overall_health": self.results.get("overall_status", "unknown"),
            "documents_indexed": db_status.get("document_count", 0),
            "retrieval_quality": quality.get("relevance_score", 0.0),
            "sample_latency_ms": self.results.get("sample_query", {}).get("latency_ms", 0),
            "batch_success_rate": round(
                batch.get("successful", 0) / batch.get("queries_count", 1) * 100, 1
            ) if batch.get("queries_count", 0) > 0 else 0,
        }

        return summary
