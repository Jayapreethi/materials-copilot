#!/usr/bin/env python
"""Run CO2M scale-up questions against a pgvector schema and save cited answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embeddings import EmbeddingGenerator
from rag_pipeline.rag_query import build_context, ollama_chat
from vector_db.postgres_corpus_store import PostgresCorpusStore


QUESTIONS = [
    ("CO2 capture and separation", "What are the major technical and operational challenges encountered when scaling CO2 capture and separation systems from laboratory or bench scale to pilot and commercial operation?"),
    ("Carbon dioxide removal from air and ocean", "Which process limitations most strongly affect the scale-up, energy demand, and long-term performance of carbon dioxide removal systems operating on air or ocean-derived CO2?"),
    ("CO2 mineralization and durable carbonated products", "What process conditions and material characteristics determine whether CO2 mineralization can be scaled while maintaining high carbonation rates, carbon uptake, and product quality?"),
    ("CO2 conversion to chemicals, fuels, and biological products", "What factors limit the transition of CO2 conversion processes from high laboratory conversion or selectivity to continuous production at pilot and industrial scales?"),
    ("CO2 conditioning, transport, and geologic storage", "What engineering considerations control the reliable scale-up of integrated CO2 conditioning, transport, injection, and long-term geologic storage systems?"),
    ("Hydrogen production", "What are the principal efficiency, durability, materials, and balance-of-plant challenges that emerge when hydrogen-production technologies are scaled to industrial capacity?"),
    ("Hydrogen storage, transport, carriers, and use", "What technical factors determine the safety, efficiency, and economic feasibility of storing, transporting, and using hydrogen at large scale?"),
    ("Ammonia, methanol, and power-to-X fuels", "How does increasing production scale affect process integration, energy efficiency, feedstock requirements, and product quality in power-to-X fuel manufacturing?"),
    ("Biofuels, waste-to-fuels, and biorefineries", "What feedstock, reaction, separation, and process-integration challenges most commonly prevent biofuel and waste-to-fuel processes from reproducing laboratory performance at commercial scale?"),
    ("Batteries and electrochemical energy storage", "Which manufacturing and process-control parameters become most important for maintaining electrochemical performance, safety, and product consistency as battery production is scaled up?"),
    ("Long-duration, thermal, and mechanical energy storage", "What technical and system-level constraints determine whether long-duration, thermal, or mechanical energy-storage technologies can be successfully scaled to grid-relevant capacities?"),
    ("Industrial heat, electrification, and waste heat recovery", "What process-integration and equipment challenges must be addressed when industrial heating processes are electrified or waste heat recovery is implemented at plant scale?"),
    ("Cement, lime, concrete, and mineral-processing decarbonization", "What process changes are required to reduce carbon emissions from cement, lime, concrete, and mineral processing without compromising throughput, product quality, or plant reliability?"),
    ("Iron, steel, and metals decarbonization", "What are the main process, energy, materials-handling, and product-quality challenges associated with scaling low-carbon technologies for iron, steel, and metals production?"),
    ("Chemicals, refining, and circular carbon systems", "Which process-integration, separation, catalyst, and feedstock challenges determine the scalability of lower-carbon chemical, refining, and circular-carbon processes?"),
    ("Critical minerals and energy-materials manufacturing", "What processing, separation, purification, and quality-control challenges arise when critical-mineral and energy-material manufacturing is scaled from laboratory processes to industrial production?"),
    ("Water-energy and desalination technologies", "What operational factors control energy consumption, fouling, recovery, water quality, and reliability when water-treatment and desalination technologies are scaled to industrial operation?"),
]


def write_report(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="co2m_unique")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-generate", action="store_true", help="Retrieve source chunks without calling Ollama")
    parser.add_argument("--output", type=Path, default=Path("co2m/metadata/scaleup_query_results.json"))
    args = parser.parse_args()

    embedder = EmbeddingGenerator()
    store = PostgresCorpusStore(dsn=args.dsn, schema=args.schema)
    vectors = embedder.embed([question for _, question in QUESTIONS]).tolist()
    results: list[dict] = []
    for number, ((topic, question), vector) in enumerate(zip(QUESTIONS, vectors), 1):
        chunks = store.search(vector, top_k=args.top_k)
        sources = [
            {
                "filename": chunk["filename"],
                "page": chunk["page_start"],
                "distance": chunk["distance"],
                "excerpt": chunk["content"][:800],
            }
            for chunk in chunks
        ]
        result = {"number": number, "topic": topic, "question": question, "sources": sources}
        if not args.no_generate:
            try:
                result["answer"] = ollama_chat(args.ollama_url, args.model, question, build_context(chunks), args.timeout)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        write_report(args.output, results)
        print(f"completed={number}/{len(QUESTIONS)} topic={topic}", flush=True)


if __name__ == "__main__":
    main()