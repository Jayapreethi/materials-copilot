"""Dashboard RAG adapter using Postgres retrieval and optional local Transformers inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class RAGQueryService:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str | None = None,
        collection_name: str | None = None,
        device: str = "cpu",
        dsn: str | None = None,
        schema: str = "public",
    ) -> None:
        from embeddings.embeddings import EmbeddingGenerator
        from vector_db.postgres_corpus_store import PostgresCorpusStore

        self.embedder = EmbeddingGenerator(model_name=embedding_model, device=device)
        self.store = PostgresCorpusStore(dsn=dsn, schema=schema)
        self.backend = os.getenv("CO2M_LLM_BACKEND", "transformers").strip().lower()
        self.model_path = os.getenv("CO2M_LLM_MODEL", "").strip()
        self._llm = None

    def get_stats(self) -> dict[str, int]:
        stats = self.store.stats()
        return {
            "total_documents": stats["unique_documents"],
            "total_chunks": stats["chunks"],
            "embedded_chunks": stats["embedded_chunks"],
            "unique_documents": stats["unique_documents"],
            "unique_chunks": stats["unique_chunks"],
            "unique_embedded_chunks": stats["unique_embedded_chunks"],
            "all_documents": stats["documents"],
        }

    def get_llm_status(self) -> dict[str, Any]:
        configured = self.backend == "transformers" and bool(self.model_path)
        return {
            "available": configured,
            "provider": "local Transformers" if configured else "local Transformers (not configured)",
            "model": Path(self.model_path).name if self.model_path else None,
            "auto_detected": False,
        }

    def _get_llm(self):
        if self.backend != "transformers":
            raise RuntimeError("Dashboard generation supports only the local Transformers backend")
        if not self.model_path:
            raise RuntimeError("Set CO2M_LLM_MODEL to a local Llama checkpoint directory")
        if self._llm is None:
            from local_llm import LocalTransformersModel

            self._llm = LocalTransformersModel(
                self.model_path,
                device_map=os.getenv("CO2M_LLM_DEVICE_MAP", "auto"),
                torch_dtype=os.getenv("CO2M_LLM_DTYPE", "auto"),
                local_files_only=True,
            )
        return self._llm

    def _retrieve(
        self,
        question: str,
        top_k: int,
        concept_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        embedding = self.embedder.embed_single(question).tolist()
        search_limit = top_k * 5 if concept_filter else top_k
        chunks = self.store.search(embedding, top_k=search_limit, unique_only=True)
        if concept_filter:
            chunks = [
                chunk for chunk in chunks
                if concept_filter in (chunk.get("chunk_metadata") or {}).get("concepts", [])
            ]
        results = []
        for rank, chunk in enumerate(chunks[:top_k], 1):
            distance = float(chunk.get("distance") or 0)
            results.append({
                "rank": rank,
                "filename": chunk.get("filename") or "Unknown",
                "page_number": chunk.get("page_start"),
                "text": chunk.get("content") or "",
                "similarity": max(0.0, min(1.0, 1.0 - distance)),
                "distance": distance,
                "title": chunk.get("title"),
            })
        return results

    @staticmethod
    def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
        similarities = [result["similarity"] for result in results]
        return {
            "total_results": len(results),
            "unique_documents": len({result["filename"] for result in results}),
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "materials_found": 0,
        }

    @staticmethod
    def _context(results: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{result['rank']}] ({result['filename']}, p.{result.get('page_number') or 'N/A'})\n"
            f"{result['text']}"
            for result in results
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        concept_filter: str | None = None,
        include_context: bool = True,
    ) -> dict[str, Any]:
        results = self._retrieve(question, top_k, concept_filter)
        response: dict[str, Any] = {
            "num_results": len(results),
            "results": results,
            "context_summary": self._summary(results),
        }
        if include_context:
            response["context"] = self._context(results)
        return response

    def generate_with_llm(
        self,
        question: str,
        top_k: int = 5,
        concept_filter: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        results = self.query(question, top_k, concept_filter, include_context=True)
        if not results["results"]:
            results["llm_response"] = "No relevant source excerpts were found."
            return results

        system_prompt = (
            "You are a scientific literature assistant. Answer only from the provided source excerpts. "
            "Cite claims with the excerpt numbers, such as [1]. State clearly when evidence is insufficient."
        )
        local_model = self._get_llm()
        results["llm_response"] = local_model.generate_text(
            system_prompt,
            f"Sources:\n{results['context']}\n\nQuestion: {question}",
            max_new_tokens=max_tokens,
        )
        results["llm_provider"] = "local Transformers"
        results["llm_model"] = Path(self.model_path).name
        results["llm_usage"] = {}
        results["llm_sources"] = [
            {
                "file": result["filename"],
                "page": result["page_number"],
                "similarity": result["similarity"],
            }
            for result in results["results"]
        ]
        return results