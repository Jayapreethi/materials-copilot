"""
embeddings.py
-------------
Embedding generation for text chunks using sentence-transformers.
Supports batch processing and GPU acceleration.
"""

import logging
import re
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _sanitize(text: str) -> str:
    """Strip lone surrogate code points that crash the Rust tokenizer."""
    if not isinstance(text, str):
        text = str(text)
    text = _SURROGATE_RE.sub("", text)
    return text.encode("utf-8", "ignore").decode("utf-8") or " "


class EmbeddingGenerator:
    """Generate embeddings for text using a sentence-transformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model ID for sentence-transformers
            device: "cpu" or "cuda"
            batch_size: Number of texts to embed simultaneously
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        logger.debug(f"Embedding {len(texts)} text(s)")
        texts = [_sanitize(text) for text in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            1D numpy array of length dimension
        """
        return self.embed([text])[0]
