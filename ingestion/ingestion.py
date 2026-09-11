"""
ingestion.py
-----------
Incremental ingestion pipeline for PDFs.
Handles deduplication, embedding generation, and vector store updates.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from embeddings.embeddings import EmbeddingGenerator
from vector_db.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionConfig:
    """Load and manage ingestion configuration."""

    def __init__(self, config_path: Path):
        with open(config_path, encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

    def get(self, key: str, default: Any = None) -> Any:
        """Nested config access: get('embedding.model')"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value is not None else default


class IngestionPipeline:
    """
    Manage incremental ingestion of PDFs into vector store.
    """

    def __init__(
        self,
        config_path: Path = Path("configs/vector_db.yaml"),
        artifacts_dir: Path = Path("co2m/artifacts"),
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config_path: Path to vector_db.yaml config
            artifacts_dir: Directory containing pipeline artifacts
        """
        self.config = IngestionConfig(config_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.manifest_path = self.artifacts_dir / "ingestion_manifest.json"

        # Initialize embedding generator
        self.embedding_gen = EmbeddingGenerator(
            model_name=self.config.get("embedding.model"),
            device=self.config.get("embedding.device", "cpu"),
            batch_size=self.config.get("embedding.batch_size", 32),
            normalize=self.config.get("embedding.normalize_embeddings", True),
        )

        # Initialize vector store
        self.vector_store = VectorStore(
            persist_directory=self.config.get("vector_store.persist_directory"),
            collection_name=self.config.get("vector_store.collection_name"),
            distance_metric=self.config.get("vector_store.chroma.distance_metric", "cosine"),
        )

        self.manifest = self._load_manifest()

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

    def ingest_artifacts(
        self,
        chunks_df: pd.DataFrame,
        concepts_df: pd.DataFrame,
        documents_df: pd.DataFrame,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest chunks from preprocessed artifacts into vector store.

        Args:
            chunks_df: DataFrame with columns: chunk_id, doc_id, chunk_text, filename, page_number
            concepts_df: DataFrame with columns: chunk_id, concept_name, confidence_score
            documents_df: DataFrame with document metadata
            force_reindex: Force re-indexing even if file exists

        Returns:
            Dict with ingestion stats
        """
        logger.info(f"Starting ingestion of {len(chunks_df)} chunk(s)")

        # Check for duplicates
        if not force_reindex and self.config.get("indexing.auto_deduplicate", True):
            chunks_df = self._filter_new_chunks(chunks_df)

        if chunks_df.empty:
            logger.warning("No new chunks to ingest")
            return {"ingested": 0, "skipped": 0}

        # Prepare data for embedding
        texts = chunks_df["chunk_text"].tolist()
        chunk_ids = [f"chunk_{cid}" for cid in chunks_df["chunk_id"]]

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} chunk(s)...")
        embeddings = self.embedding_gen.embed(texts)

        # Prepare metadata
        metadatas = []
        for _, row in chunks_df.iterrows():
            chunk_id = row["chunk_id"]
            # Get concepts for this chunk
            chunk_concepts = concepts_df[concepts_df["chunk_id"] == chunk_id]
            concept_names = chunk_concepts["concept_name"].tolist()
            max_confidence = (
                chunk_concepts["confidence_score"].max()
                if not chunk_concepts.empty
                else 0.0
            )

            metadata = {
                "chunk_id": int(chunk_id),
                "doc_id": int(row["doc_id"]),
                "filename": str(row["filename"]),
                "page_number": int(row["page_number"]),
                "concept_names": ",".join(concept_names) if concept_names else "unassigned",
                "max_confidence": float(max_confidence),
            }

            # Add document metadata
            doc_match = documents_df[documents_df["doc_id"] == row["doc_id"]]
            if not doc_match.empty:
                doc_row = doc_match.iloc[0]
                metadata["title"] = str(doc_row.get("title", ""))
                metadata["authors"] = str(doc_row.get("authors", ""))
                year_val = doc_row.get("year")
                if pd.notna(year_val):
                    metadata["year"] = int(year_val)

            # Clean metadata for Chroma compliance
            metadata = self._clean_metadata(metadata)
            metadatas.append(metadata)

        # Add to vector store
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=chunk_ids,
        )

        # Persist
        self.vector_store.persist()

        # Update manifest
        self.manifest["last_ingestion"] = datetime.now(timezone.utc).isoformat()
        self.manifest["total_chunks"] = self.vector_store.collection.count()
        self.manifest["embedding_model"] = self.config.get("embedding.model")
        self.manifest["embedding_dimension"] = self.embedding_gen.dimension
        self._save_manifest()

        stats = {
            "ingested": len(texts),
            "embedding_model": self.config.get("embedding.model"),
            "collection_stats": self.vector_store.get_stats(),
        }
        logger.info(f"Ingestion complete. Stats: {stats}")
        return stats

    def _filter_new_chunks(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter chunks that are already in vector store.
        Uses filename and chunk_id to detect duplicates.
        """
        logger.debug("Filtering for new chunks...")
        indexed_files = self.manifest.get("indexed_files", {})

        new_chunks = []
        for _, row in chunks_df.iterrows():
            filename = row["filename"]
            chunk_id = row["chunk_id"]
            file_hash = self.manifest.get("file_hashes", {}).get(filename)

            # If file hasn't been indexed before, include it
            if filename not in indexed_files:
                new_chunks.append(row)
                continue

            # If already indexed, skip unless force
            logger.debug(f"Skipping previously indexed file: {filename}")

        if new_chunks:
            return pd.DataFrame(new_chunks).reset_index(drop=True)
        return pd.DataFrame()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load ingestion manifest or create empty one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"indexed_files": {}, "file_hashes": {}}

    def _save_manifest(self) -> None:
        """Save ingestion manifest."""
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2)
        logger.debug(f"Manifest saved to {self.manifest_path}")
