"""Ranker component for document reranking using Qwen3-Reranker-4B."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.adapters.reranker_adapter import Qwen3RerankerAdapter, RerankerConfig

logger = logging.getLogger(__name__)


@dataclass
class RankerConfig:
    """Configuration for the Ranker component."""

    relevance_threshold: float = 0.5
    batch_size: int = 8
    top_k: int = 10

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.relevance_threshold <= 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


class Ranker:
    """Ranks and filters documents based on relevance to query."""

    def __init__(self, config: RankerConfig | None = None):
        """Initialize the Ranker.

        Args:
            config: Ranker configuration
        """
        self.config = config or RankerConfig()

        # Initialize reranker adapter
        reranker_config = RerankerConfig(batch_size=self.config.batch_size)
        self.reranker = Qwen3RerankerAdapter(reranker_config)

        logger.info(
            f"Ranker initialized with threshold={self.config.relevance_threshold}"
        )

    def rank_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_field: str = "content",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank and filter documents by relevance.

        Args:
            query: Query text
            documents: List of document dictionaries
            text_field: Field name containing document text
            top_k: Number of top documents to return (overrides config)

        Returns:
            List of ranked documents with rerank scores
        """
        if not documents:
            return []

        start_time = time.time()
        top_k = top_k if top_k is not None else self.config.top_k

        logger.info(f"Starting ranking of {len(documents)} documents")

        try:
            # Extract text content
            texts = []
            for doc in documents:
                if text_field not in doc:
                    raise KeyError(f"Document missing required field '{text_field}'")
                texts.append(doc[text_field])

            # Rerank documents
            rerank_results = self.reranker.rerank(
                query=query,
                documents=texts,
                batch_size=self.config.batch_size,
                top_k=top_k,
            )

            # Build ranked documents with scores
            ranked_docs = []
            rank = 1

            for result in rerank_results:
                # Apply relevance threshold
                if result.score < self.config.relevance_threshold:
                    logger.debug(
                        f"Filtering out document with score "
                        f"{result.score:.4f} < threshold"
                    )
                    continue

                # Get original document and add ranking info
                doc = documents[result.original_index].copy()
                doc["rerank_score"] = result.score
                doc["rerank_rank"] = rank
                ranked_docs.append(doc)
                rank += 1

            elapsed_time = time.time() - start_time
            logger.info(
                f"Ranking completed in {elapsed_time:.2f}s. "
                f"Kept {len(ranked_docs)}/{len(documents)} documents above threshold"
            )

            return ranked_docs

        except Exception as e:
            logger.error(f"Ranking failed: {e!s}")
            raise

    def get_statistics(self) -> dict[str, Any]:
        """Get ranker statistics.

        Returns:
            Dictionary containing ranker and reranker statistics
        """
        stats = {
            "config": {
                "relevance_threshold": self.config.relevance_threshold,
                "batch_size": self.config.batch_size,
                "default_top_k": self.config.top_k,
            },
            "reranker_stats": self.reranker.get_stats(),
        }
        return stats
