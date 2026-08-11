#!/usr/bin/env python
"""
Enhanced Vector Database Ingestion Pipeline
Ingests CO2M artifacts with enriched structured metadata

Run: python ingest_enhanced.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yaml

from embeddings.embeddings import EmbeddingGenerator
from vector_db.vector_store import VectorStore
from metadata_enricher import MetadataEnricher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedIngestionPipeline:
    """Ingestion pipeline with rich metadata enrichment."""

    def __init__(self, config_path: str = "./configs/vector_db.yaml"):
        self.config = self._load_config(config_path)
        self.embedding_gen = EmbeddingGenerator(
            model_name=self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=self.config.get("device", "cpu"),
        )
        self.vector_store = VectorStore()
        self.enricher = MetadataEnricher()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Clean metadata for Chroma compliance.
        - Remove None values
        - Remove empty strings
        - Remove empty lists/dicts
        - Convert non-serializable types to strings
        """
        cleaned = {}
        for key, value in metadata.items():
            # Skip None, empty strings, empty lists, empty dicts
            if value is None or value == "" or value == [] or value == {}:
                continue
            
            # Convert to appropriate types
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (list, dict)):
                # Skip complex types not supported by Chroma
                if isinstance(value, list) and value:
                    # Convert list to comma-separated string if not empty
                    cleaned[key] = ",".join(str(v) for v in value)
                continue
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        
        return cleaned

    def ingest_enhanced(
        self,
        chunks_df: pd.DataFrame,
        documents_df: pd.DataFrame,
        concepts_df: pd.DataFrame = None,
    ) -> dict:
        """
        Ingest chunks with enriched metadata.

        Args:
            chunks_df: Chunks dataframe
            documents_df: Documents dataframe
            concepts_df: Concepts dataframe (optional)

        Returns:
            Ingestion statistics
        """
        logger.info(f"Starting enhanced ingestion of {len(chunks_df)} chunks...")

        texts = []
        embeddings_list = []
        metadatas = []
        ids = []
        batch_size = 100

        # Process each chunk
        for idx, (_, chunk_row) in enumerate(chunks_df.iterrows()):
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processing chunk {idx + 1}/{len(chunks_df)}...")

            chunk_id = int(chunk_row["chunk_id"])
            doc_id = int(chunk_row["doc_id"])
            chunk_text = str(chunk_row["chunk_text"])
            page_number = int(chunk_row["page_number"])

            # Get document info
            doc_match = documents_df[documents_df["doc_id"] == doc_id]
            filename = (
                str(doc_match.iloc[0]["filename"])
                if not doc_match.empty
                else f"doc_{doc_id}"
            )

            # Enrich metadata
            enriched_meta = self.enricher.enrich(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=chunk_id,
                filename=filename,
                page_number=page_number,
            )

            # Add concepts if available
            if concepts_df is not None:
                chunk_concepts = concepts_df[
                    concepts_df["chunk_id"] == chunk_id
                ]
                concept_names = chunk_concepts["concept_name"].tolist()
                enriched_meta["concepts"] = ",".join(concept_names) if concept_names else "unassigned"
                enriched_meta["max_confidence"] = (
                    float(chunk_concepts["confidence_score"].max())
                    if not chunk_concepts.empty
                    else 0.0
                )

            # Clean metadata for Chroma compliance
            enriched_meta = self._clean_metadata(enriched_meta)

            # Collect for batch processing
            texts.append(chunk_text)
            metadatas.append(enriched_meta)
            ids.append(f"paper_{doc_id:03d}_chunk_{chunk_id:05d}")

            # Batch process embeddings every N chunks
            if (idx + 1) % batch_size == 0 or idx == len(chunks_df) - 1:
                # Only embed the new batch that hasn't been embedded yet
                start_idx = len(embeddings_list)
                texts_to_embed = texts[start_idx:]
                
                if texts_to_embed:
                    logger.info(f"    Generating embeddings for batch {(idx + 1) // batch_size}...")
                    batch_embeddings = self.embedding_gen.embed(texts_to_embed)
                    embeddings_list.extend(batch_embeddings.tolist())

        # Verify all texts have embeddings
        if len(embeddings_list) != len(texts):
            logger.warning(f"Embedding mismatch: {len(embeddings_list)} embeddings for {len(texts)} texts")
            logger.info(f"Generating missing embeddings...")
            remaining_embeddings = self.embedding_gen.embed(texts[len(embeddings_list):])
            embeddings_list.extend(remaining_embeddings.tolist())

        # Add all to vector store
        logger.info(f"Adding {len(texts)} documents to vector store...")
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("✓ Ingestion complete")

        return {
            "ingested": len(texts),
            "embedding_model": self.embedding_gen.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def validate_metadata(self, metadatas: list) -> dict:
        """Validate enriched metadata."""
        stats = {
            "total": len(metadatas),
            "with_material": sum(1 for m in metadatas if "material_name" in m),
            "with_measurement": sum(1 for m in metadatas if "measurement" in m),
            "with_conditions": sum(1 for m in metadatas if "temperature_K" in m),
            "with_properties": sum(1 for m in metadatas if "surface_area_m2_g" in m),
        }

        logger.info(f"\nMetadata Statistics:")
        logger.info(f"  Total chunks: {stats['total']}")
        logger.info(f"  With material info: {stats['with_material']} ({100*stats['with_material']/stats['total']:.1f}%)")
        logger.info(f"  With measurements: {stats['with_measurement']} ({100*stats['with_measurement']/stats['total']:.1f}%)")
        logger.info(f"  With conditions: {stats['with_conditions']} ({100*stats['with_conditions']/stats['total']:.1f}%)")
        logger.info(f"  With properties: {stats['with_properties']} ({100*stats['with_properties']/stats['total']:.1f}%)")

        return stats


def main():
    """Main ingestion script."""
    print("=" * 70)
    print("ENHANCED VECTOR DATABASE INGESTION")
    print("=" * 70)

    # Load artifacts
    logger.info("[1] Loading artifacts...")
    artifacts_path = Path("./co2m/artifacts")

    chunks_df = pd.read_parquet(artifacts_path / "chunks.parquet")
    documents_df = pd.read_parquet(artifacts_path / "documents.parquet")
    concepts_df = pd.read_parquet(artifacts_path / "concepts.parquet")

    logger.info(f"  Loaded {len(chunks_df)} chunks from {len(documents_df)} documents")

    # Initialize pipeline
    logger.info("\n[2] Initializing pipeline...")
    pipeline = EnhancedIngestionPipeline()

    # Ingest with enrichment
    logger.info("\n[3] Ingesting with metadata enrichment...")
    stats = pipeline.ingest_enhanced(
        chunks_df=chunks_df,
        documents_df=documents_df,
        concepts_df=concepts_df,
    )

    logger.info(f"  ✓ Ingested {stats['ingested']} documents")
    logger.info(f"  Model: {stats['embedding_model']}")

    # Validate metadata
    logger.info("\n[4] Validating enriched metadata...")
    # Get sample to validate
    sample_results = pipeline.vector_store.collection.get(limit=100, include=["metadatas"])
    pipeline.validate_metadata(sample_results["metadatas"])

    # Save manifest
    logger.info("\n[5] Saving ingestion manifest...")
    manifest = {
        "ingestion_timestamp": stats["timestamp"],
        "total_chunks": stats["ingested"],
        "embedding_model": stats["embedding_model"],
        "format": "enhanced_structured",
        "metadata_fields": [
            "document_id",
            "section",
            "page",
            "material_name",
            "material_class",
            "material_formula",
            "measurement",
            "value",
            "unit",
            "temperature_K",
            "pressure_bar",
            "humidity_percent",
            "surface_area_m2_g",
            "pore_volume_cm3_g",
            "pore_size_nm",
            "method",
            "gas",
            "source_type",
            "concepts",
        ]
    }

    with open("./co2m/artifacts/enhanced_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  ✓ Manifest saved")

    print("\n" + "=" * 70)
    print("✓ ENHANCED INGESTION COMPLETE")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        exit(1)
