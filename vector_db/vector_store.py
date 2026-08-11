"""
vector_store.py
---------------
Vector database management using Chroma.
Handles ingestion, retrieval, and incremental updates.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper around Chroma vector database for CO2M corpus."""

    def __init__(
        self,
        persist_directory: str = "./vector_db/data",
        collection_name: str = "co2m_corpus",
        distance_metric: str = "cosine",
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Path to persist Chroma data
            collection_name: Name of the collection
            distance_metric: "cosine" (default), "l2", etc.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        logger.info(
            f"Initializing Chroma client with persist_dir: {self.persist_directory}"
        )
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info(f"Collection '{collection_name}' ready. Current size: {self.collection.count()}")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """
        Add documents with embeddings to the collection.

        Args:
            texts: List of text chunks
            embeddings: Pre-computed embeddings (from EmbeddingGenerator)
            metadatas: List of metadata dicts (doc_id, chunk_id, filename, page_number, etc.)
            ids: Unique IDs for each chunk
        """
        if not texts:
            logger.warning("No texts to add")
            return

        logger.info(f"Adding {len(texts)} document(s) to collection")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        logger.info(f"Collection size after add: {self.collection.count()}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k most similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional Chroma where filter (e.g., {"concept_name": {"$eq": "Climate modeling"}})

        Returns:
            Dict with 'ids', 'distances', 'metadatas', 'documents'
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
        }

    def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks belonging to a file (for re-indexing).

        Args:
            filename: Name of the PDF file

        Returns:
            Number of deleted items
        """
        logger.info(f"Deleting all chunks from file: {filename}")
        where = {"filename": {"$eq": filename}}
        ids_to_delete = self.collection.get(where=where)["ids"]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunk(s)")
        return len(ids_to_delete)

    def persist(self) -> None:
        """Persist collection to disk."""
        logger.info("Persisting collection to disk")
        # PersistentClient automatically persists data, no explicit persist needed

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "distance_metric": self.distance_metric,
            "persist_directory": str(self.persist_directory),
        }

    @staticmethod
    def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute SHA256 hash of a file for deduplication.

        Args:
            file_path: Path to file
            chunk_size: Read chunk size

        Returns:
            Hex hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
