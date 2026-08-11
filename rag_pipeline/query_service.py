"""
Query Service - Business logic layer for RAG queries and context generation.
"""

import logging
from typing import Any, Dict, List, Optional

from .retriever import RAGRetriever
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGQueryService:
    """Service layer for RAG queries, context generation, and result formatting."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "./vector_db/data",
        collection_name: str = "co2m_corpus",
        device: str = "cpu",
        llm_provider: str = "auto",
        llm_model: str = None,
        llm_api_key: str = None,
    ):
        """
        Initialize query service.

        Args:
            embedding_model: HuggingFace model ID for embeddings
            vector_store_path: Path to Chroma vector store
            collection_name: Name of the collection
            device: "cpu" or "cuda" for embedding computation
            llm_provider: LLM provider ('auto', 'openai', 'anthropic', 'ollama', 'huggingface', 'together')
                         'auto' auto-detects available free providers first, then paid
            llm_model: LLM model name (optional, auto-selected per provider if not specified)
            llm_api_key: API key for LLM provider (auto-detected from env vars)
        """
        self.retriever = RAGRetriever(
            embedding_model=embedding_model,
            vector_store_path=vector_store_path,
            collection_name=collection_name,
            device=device,
        )
        
        # Initialize LLM service
        self.llm = None
        try:
            self.llm = LLMService(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
            )
        except Exception as e:
            logger.warning(f"LLM service not available: {e}")


    def query(
        self,
        question: str,
        top_k: int = 5,
        concept_filter: Optional[str] = None,
        include_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a RAG query with context.

        Args:
            question: User question/query
            top_k: Number of retrieval results
            concept_filter: Optional concept to filter by
            include_context: Whether to format context for LLM

        Returns:
            Query response with retrieved documents and formatted context
        """
        logger.info(f"Query: {question}")

        # Retrieve documents
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            concept_filter=concept_filter,
        )

        response = {
            "question": question,
            "concept_filter": concept_filter,
            "num_results": retrieval_results["total_results"],
            "results": retrieval_results["results"],
        }

        if include_context:
            response["context"] = self._format_context(retrieval_results["results"])
            response["context_summary"] = self._generate_summary(retrieval_results["results"])

        return response

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieval results as context for LLM.

        Args:
            results: List of retrieval results

        Returns:
            Formatted context string
        """
        context_parts = []
        for result in results:
            part = (
                f"[Source {result['rank']}: {result['filename']} (Page {result['page_number']}), "
                f"Similarity: {result['similarity']}]\n"
                f"{result['text']}\n"
            )
            context_parts.append(part)

        return "\n".join(context_parts)

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary of retrieval results.

        Args:
            results: List of retrieval results

        Returns:
            Summary dict with key statistics
        """
        unique_files = set(r["filename"] for r in results)
        material_results = [r for r in results if r.get("material_name")]
        co2_results = [r for r in results if r.get("co2_uptake_mmol_g")]

        summary = {
            "total_results": len(results),
            "unique_documents": len(unique_files),
            "documents": list(unique_files),
            "avg_similarity": round(sum(r["similarity"] for r in results) / len(results), 4) if results else 0,
            "materials_found": len(material_results),
            "co2_data_found": len(co2_results),
        }

        if co2_results:
            co2_values = [float(r["co2_uptake_mmol_g"]) for r in co2_results if r["co2_uptake_mmol_g"]]
            if co2_values:
                summary["co2_uptake_stats"] = {
                    "min": min(co2_values),
                    "max": max(co2_values),
                    "avg": sum(co2_values) / len(co2_values),
                }

        return summary

    def search_by_topic(
        self,
        topic: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Search documents by topic.

        Args:
            topic: Topic/keyword to search
            top_k: Number of results

        Returns:
            Search results organized by topic
        """
        logger.info(f"Topic search: {topic}")
        results = self.retriever.retrieve(query=topic, top_k=top_k)

        return {
            "topic": topic,
            "results": results["results"],
            "count": len(results["results"]),
        }

    def get_related_documents(
        self,
        document_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Find documents related to a given text.

        Args:
            document_text: Text to find related documents for
            top_k: Number of related documents to return

        Returns:
            Related documents
        """
        logger.info("Finding related documents")
        results = self.retriever.retrieve(query=document_text, top_k=top_k)

        return {
            "query_length": len(document_text),
            "related_documents": results["results"],
            "count": len(results["results"]),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.retriever.get_stats()

    def generate_with_llm(
        self,
        question: str,
        top_k: int = 5,
        concept_filter: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate LLM response using RAG context.

        Args:
            question: User question
            top_k: Number of retrieval results
            concept_filter: Optional concept filter
            temperature: LLM sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Combined RAG+LLM response with retrieved context and generated answer
        """
        # First, retrieve relevant documents
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            concept_filter=concept_filter,
        )

        response = {
            "question": question,
            "concept_filter": concept_filter,
            "retrieved_results": retrieval_results["results"],
            "results": retrieval_results["results"],  # For dashboard compatibility
            "num_results": retrieval_results["total_results"],
            "context": self._format_context(retrieval_results["results"]),
            "context_summary": self._generate_summary(retrieval_results["results"]),
            "llm_response": None,
            "llm_available": False,
            "error": None,
        }

        # Generate LLM response if available
        if self.llm and self.llm.is_available():
            try:
                response["llm_available"] = True
                llm_result = self.llm.generate_with_rag(
                    question=question,
                    retrieved_results=retrieval_results["results"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response["llm_response"] = llm_result.get("response", "")
                response["llm_sources"] = llm_result.get("sources", [])
                response["llm_usage"] = llm_result.get("usage", {})
                response["llm_model"] = llm_result.get("model", "")
                response["llm_provider"] = llm_result.get("provider", "")

                # If response is empty (e.g. provider failed silently), synthesize from context
                if not response["llm_response"]:
                    response["llm_response"] = self._synthesize_fallback(
                        question, response["context"]
                    )
                    response["llm_provider"] = response["llm_provider"] or "fallback"
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response["error"] = str(e)
                # Still synthesize a structured response from retrieved context
                try:
                    response["llm_response"] = self._synthesize_fallback(
                        question, response["context"]
                    )
                    response["llm_provider"] = "fallback"
                    response["llm_available"] = True
                    response["error"] = None
                except Exception as fe:
                    logger.error(f"Fallback synthesis failed: {fe}")
        else:
            response["error"] = "LLM service not available"

        return response

    def _synthesize_fallback(self, question: str, context: str) -> str:
        """Synthesize a structured markdown response directly from RAG context."""
        import re

        source_blocks = re.split(r'\[Source \d+[^\]]*\]', context)
        source_headers = re.findall(r'\[Source \d+[^\]]*\]', context)

        def is_reference_dump(text: str) -> bool:
            ref_tags = len(re.findall(r'\[\d+\]', text))
            words = max(len(text.split()), 1)
            return (ref_tags / words) >= 0.03

        def extract_content(block: str) -> str:
            lines = block.split('\n')
            return ' '.join(l for l in lines if not l.strip().startswith('File:')).strip()

        def extract_file_info(block: str) -> str:
            m = re.search(r'File:\s*(.+?),\s*Page:\s*(\S+)', block)
            return f"{m.group(1)} (Page {m.group(2)})" if m else None

        # Collect clean passages and unique filenames
        clean_passages, seen_keys, file_infos, seen_fnames = [], set(), [], set()
        for header, block in zip(source_headers, source_blocks[1:]):
            block = block.strip()
            if not block:
                continue
            fi = extract_file_info(block)
            if fi:
                fname = fi.split(' (Page')[0]
                if fname not in seen_fnames:
                    seen_fnames.add(fname)
                    file_infos.append(fi)
            content = extract_content(block)
            key = content[:120]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            is_header = bool(re.search(r'\baDept\b|\bCorresponding author\b|\bNumber of pages\b', content))
            if not is_reference_dump(content) and not is_header and len(content) > 80:
                clean_passages.append(content)

        # If still nothing, strip inline cites from raw blocks
        if not clean_passages:
            for _, block in zip(source_headers, source_blocks[1:]):
                content = re.sub(r'\[\d+\]', '', extract_content(block)).strip()
                if len(content) > 80 and content[:120] not in seen_keys:
                    seen_keys.add(content[:120])
                    clean_passages.append(content)

        full_text = context
        # Materials
        mat_pats = [
            r'\bmonoethanolamine\b', r'\bdiethanolamine\b', r'\bpiperazine\b',
            r'\bMEA\b', r'\bDEA\b', r'\bMDEA\b', r'\bPZ\b',
            r'\bzeolite[s]?\b', r'\bMOF[s]?\b', r'\bactivated carbon\b',
            r'\bamine[s]?\b', r'\bsolvent[s]?\b', r'\bsorbent[s]?\b',
            r'\badsorbent[s]?\b', r'\bcalcium oxide\b', r'\bCaO\b',
            r'\bsilica[te]?\b', r'\bcement\b', r'\bbasalt\b',
            r'\bmineral[s]?\b', r'\bcarbonate[s]?\b', r'\bwollastonite\b',
            r'\bserpentine\b', r'\bolivine\b', r'\bperidotite[s]?\b',
        ]
        materials = []
        for pat in mat_pats:
            m = re.findall(pat, full_text, re.IGNORECASE)
            if m:
                c = m[0].strip().title()
                if c not in materials:
                    materials.append(c)

        pcts = sorted(set(float(p) for p in re.findall(r'(\d+(?:\.\d+)?)\s*%', full_text)
                         if 10 <= float(p) <= 100), reverse=True)

        def is_bib(s):
            return bool(re.search(r'\bvol\b|\bpp\b|\bDOE\b|\bAgency\b', s, re.I)) or len(s.split()) < 12

        pros, cons = [], []
        for sent in re.split(r'[.!?]', full_text):
            s = sent.strip()
            if not s or len(s) < 40 or is_bib(s):
                continue
            if re.search(r'\b(high|efficient|effective|commercial|proven|low cost|selective)\b', s, re.I):
                pros.append(s)
            if re.search(r'\b(energy.intensive|degradation|corros|toxic|expensive|limitation|drawback|challenge)\b', s, re.I):
                cons.append(s)

        q_lower = question.lower()
        is_material_q = any(w in q_lower for w in ['material', 'compound', 'chemical', 'what', 'which', 'type', 'kind', 'sorbent', 'solvent', 'capture'])

        out = [f"**Summary: {question}**\n"]

        if is_material_q and materials:
            out.append("**Materials / Compounds Identified in Literature:**\n")
            for mat in materials[:8]:
                out.append(f"- **{mat}**")
            out.append("")

        out.append("**Key Findings from Research Literature:**\n")
        if clean_passages:
            for p in clean_passages[:3]:
                excerpt = p[:400].replace('\n', ' ')
                out.append(f"- {excerpt}{'…' if len(p) > 400 else ''}")
        else:
            out.append("- See source excerpts in Detailed Results below.")
        out.append("")

        if pros:
            out.append("**Advantages noted:**\n")
            for s in pros[:3]:
                out.append(f"- {s.strip()}")
            out.append("")
        if cons:
            out.append("**Challenges / Limitations noted:**\n")
            for s in cons[:3]:
                out.append(f"- {s.strip()}")
            out.append("")

        if pcts:
            out.append("**Quantitative Data:**\n")
            out.append(f"- Efficiencies up to **{pcts[0]:.0f}%** mentioned in the literature")
            out.append("")

        if file_infos:
            out.append("**Sources consulted:**\n")
            for fi in file_infos[:5]:
                out.append(f"- {fi}")
            out.append("")

        out.append("*See Detailed Results below for full source excerpts.*")
        return "\n".join(out)

    def get_llm_status(self) -> Dict[str, Any]:
        """Get LLM service status."""
        if not self.llm:
            return {
                "available": False,
                "message": "LLM service not initialized",
            }
        return self.llm.get_status()
