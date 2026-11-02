"""Dashboard-facing query service backed by Postgres/pgvector and Ollama."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embeddings.embeddings import EmbeddingGenerator
from rag_pipeline.rag_query import build_context, ollama_chat, trl_proxy_label
from vector_db.postgres_corpus_store import PostgresCorpusStore


TRL_PROXY_OPTIONS = (
    "TRL_2_3_PROXY",
    "TRL_4_5_PROXY",
    "TRL_6_7_PROXY",
    "INSUFFICIENT_EVIDENCE",
    "REVIEW_EXCLUDED",
)


class RAGQueryService:
    """Minimal dashboard adapter over the validated Postgres RAG path."""

    def __init__(
        self,
        embedding_model: str | None = None,
        vector_store_path: str | None = None,
        collection_name: str = "co2m_corpus",
        device: str = "cpu",
        dsn: str | None = None,
        schema: str = "public",
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        timeout: int = 120,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        self._embedder = EmbeddingGenerator(model_name=embedding_model, device=device) if embedding_model else EmbeddingGenerator(device=device)
        self._store = PostgresCorpusStore(dsn=dsn, schema=schema)

    def get_stats(self) -> dict[str, Any]:
        stats = self._store.stats()
        return {
            "total_documents": stats["documents"],
            "total_chunks": stats["chunks"],
            "embedded_chunks": stats["embedded_chunks"],
            "unique_documents": stats["unique_documents"],
            "backend": "postgresql+pgvector",
        }

    def get_llm_status(self) -> dict[str, Any]:
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(self.ollama_url.rstrip("/") + "/api/tags", timeout=5) as response:
                payload = json.load(response)
            available = [item.get("name", "") for item in payload.get("models", [])]
            model_available = any(
                name == self.model or name == f"{self.model}:latest" or name.startswith(self.model + ":")
                for name in available
            )
            return {
                "available": model_available,
                "provider": "ollama",
                "model": self.model,
                "auto_detected": True,
            }
        except Exception:
            return {"available": False, "provider": "ollama", "model": self.model, "auto_detected": True}

    @staticmethod
    def _format_result(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
        distance = float(chunk.get("distance", 1.0))
        proxy = (chunk.get("document_metadata") or {}).get("scale_derived_trl_proxy") or {}
        return {
            "rank": rank,
            "similarity": max(0.0, 1.0 - distance),
            "filename": chunk.get("filename") or "Unknown",
            "file": chunk.get("filename") or "Unknown",
            "page_number": chunk.get("page_start") or "?",
            "page": chunk.get("page_start") or "N/A",
            "text": chunk.get("content") or "",
            "material_name": None,
            "co2_uptake_mmol_g": None,
            "trl_proxy": proxy.get("classification"),
            "document_title": chunk.get("title"),
            "document_year": chunk.get("publication_year"),
        }

    @staticmethod
    def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {"total_results": 0, "unique_documents": 0, "avg_similarity": 0.0, "materials_found": 0}
        similarities = [item["similarity"] for item in results]
        return {
            "total_results": len(results),
            "unique_documents": len({item["filename"] for item in results}),
            "avg_similarity": sum(similarities) / len(similarities),
            "materials_found": sum(1 for item in results if item.get("material_name")),
        }

    def query(
        self,
        question: str,
        top_k: int = 5,
        concept_filter: str | None = None,
        include_context: bool = True,
        trl_proxy: Optional[str] = None,
        unique_only: bool = True,
    ) -> dict[str, Any]:
        if trl_proxy is not None and trl_proxy not in TRL_PROXY_OPTIONS:
            raise ValueError(f"Unsupported TRL proxy: {trl_proxy}")
        vector = self._embedder.embed([question])[0].tolist()
        chunks = self._store.search(vector, top_k=top_k, unique_only=unique_only, trl_proxy=trl_proxy)
        results = [self._format_result(chunk, rank) for rank, chunk in enumerate(chunks, start=1)]
        context = build_context(chunks) if include_context else ""
        return {
            "query": question,
            "num_results": len(results),
            "results": results,
            "context": context,
            "context_summary": self._summary(results),
        }

    def generate_with_llm(
        self,
        question: str,
        top_k: int = 5,
        concept_filter: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        trl_proxy: Optional[str] = None,
        unique_only: bool = True,
    ) -> dict[str, Any]:
        result = self.query(
            question,
            top_k=top_k,
            concept_filter=concept_filter,
            include_context=True,
            trl_proxy=trl_proxy,
            unique_only=unique_only,
        )
        answer = ollama_chat(self.ollama_url, self.model, question, result["context"], self.timeout)
        structured = None
        try:
            import json as _json
            parsed = _json.loads(answer)
            if isinstance(parsed, dict) and "status" in parsed:
                structured = parsed
        except (ValueError, TypeError):
            structured = None
        result.update({
            "llm_response": answer,
            "structured_answer": structured,
            "llm_provider": "ollama",
            "llm_model": self.model,
            "llm_sources": result["results"],
        })
        return result
