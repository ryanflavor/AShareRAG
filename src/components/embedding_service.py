"""Embedding Service component for text vectorization."""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Add the project root to the path to import from reserved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reserved.basic_embedding_advanced import Qwen3EmbeddingManager, clear_memory


class EmbeddingService:
    """Service for generating text embeddings using Qwen3-Embedding-4B model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        embedding_dim: int = 2560,
        batch_size: int = 32,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model
            device: Device to use ('cuda', 'cpu', or None for auto)
            embedding_dim: Dimension of embeddings
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # Initialize the embedding manager
        self.manager = Qwen3EmbeddingManager(
            model_name=model_name, device=device, embedding_dim=embedding_dim
        )

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self) -> bool:
        """
        Load the embedding model.

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Loading embedding model: {self.model_name}")
        return self.manager.load_model()

    def generate_embeddings(
        self, texts: list[str], show_progress: bool | None = None
    ) -> np.ndarray | None:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar (auto-determined if None)

        Returns:
            np.ndarray: Array of embeddings or None if failed
        """
        if not texts:
            # Return empty array with correct shape for empty input
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Check if model is loaded
        if not hasattr(self.manager, "model") or self.manager.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return None

        # Auto-determine progress display
        if show_progress is None:
            show_progress = len(texts) > 50

        # Generate embeddings
        try:
            embeddings = self.manager.encode_texts(
                texts, batch_size=self.batch_size, show_progress=show_progress
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None

    def process_documents(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Process documents with metadata and generate embeddings.

        Args:
            documents: List of document dictionaries with text and metadata

        Returns:
            List of processed documents with embeddings and metadata
        """
        if not documents:
            return []

        results = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            texts = [doc["text"] for doc in batch]

            # Generate embeddings for batch
            embeddings = self.generate_embeddings(
                texts, show_progress=len(documents) > 50
            )

            if embeddings is None:
                self.logger.error(
                    f"Failed to generate embeddings for batch starting at index {i}"
                )
                return []

            # Create result entries with metadata
            for _j, (doc, embedding) in enumerate(zip(batch, embeddings, strict=False)):
                result = {
                    "id": f"{doc['doc_id']}_{doc['chunk_index']}",
                    "text": doc["text"],
                    "vector": embedding,
                    "company_name": doc["company_name"],
                    "doc_id": doc["doc_id"],
                    "chunk_index": doc["chunk_index"],
                    "entities": doc.get("entities", []),
                    "relations": doc.get("relations", []),
                    "relations_count": len(doc.get("relations", [])),
                    "source_file": doc.get("source_file", "corpus.json"),
                    "processing_timestamp": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
                results.append(result)

        # Clear memory after processing
        clear_memory()

        return results
