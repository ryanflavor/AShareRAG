"""Fact-based Q&A pipeline for processing factual queries.

This module implements the end-to-end pipeline for fact-based question answering,
integrating vector retrieval, reranking, and answer synthesis components.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from src.adapters.llm_adapter import LLMAdapter
from src.components.answer_synthesizer import AnswerSynthesizer, AnswerSynthesizerConfig
from src.components.embedding_service import EmbeddingService
from src.components.ranker import Ranker, RankerConfig
from src.components.retriever import VectorRetriever
from src.components.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


@dataclass
class FactQAPipelineConfig:
    """Configuration for the fact-based Q&A pipeline."""

    # Retrieval settings
    retriever_top_k: int = 10

    # Reranking settings
    reranker_top_k: int = 5
    relevance_threshold: float = 0.5
    reranker_batch_size: int = 8

    # Answer synthesis settings
    answer_max_tokens: int = 500
    answer_temperature: float = 0.7
    answer_top_p: float = 0.9
    answer_language: str = "Chinese"
    include_citations: bool = True
    citation_format: str = "[{idx}]"

    # Pipeline settings
    enable_caching: bool = True
    cache_size: int = 1000


class FactQAPipeline:
    """End-to-end pipeline for fact-based question answering."""

    def __init__(
        self,
        config: FactQAPipelineConfig,
        vector_storage: VectorStorage,
        embedding_service: EmbeddingService,
        llm_adapter: LLMAdapter,
    ):
        """Initialize the fact-based Q&A pipeline.

        Args:
            config: Pipeline configuration
            vector_storage: Vector storage instance
            embedding_service: Embedding service instance
            llm_adapter: LLM adapter instance
        """
        self.config = config

        # Initialize components
        self.retriever = VectorRetriever(
            vector_storage=vector_storage,
            embedding_service=embedding_service,
            top_k=config.retriever_top_k,
        )

        self.ranker = Ranker(
            config=RankerConfig(
                top_k=config.reranker_top_k,
                relevance_threshold=config.relevance_threshold,
                batch_size=config.reranker_batch_size,
            )
        )

        self.synthesizer = AnswerSynthesizer(
            config=AnswerSynthesizerConfig(
                max_input_tokens=3000,
                max_output_tokens=config.answer_max_tokens,
                temperature=config.answer_temperature,
                top_p=config.answer_top_p,
                answer_language=config.answer_language,
                include_citations=config.include_citations,
                citation_format=config.citation_format,
            ),
            llm_adapter=llm_adapter,
        )

        # Setup caching if enabled
        self._cache = None
        if config.enable_caching:
            # Use LRU cache for query results
            self._cache = {}
            self._cache_stats = {"hits": 0, "misses": 0}

        # Statistics
        self._total_queries = 0

        logger.info(
            f"FactQAPipeline initialized with config: "
            f"retriever_k={config.retriever_top_k}, "
            f"reranker_k={config.reranker_top_k}, "
            f"language={config.answer_language}"
        )

    async def process_query(
        self, query: str, synthesis_prompt: str | None = None
    ) -> dict[str, Any]:
        """Process a fact-based query through the pipeline.

        Args:
            query: The user's query
            synthesis_prompt: Optional custom prompt for answer synthesis

        Returns:
            Dictionary containing:
                - answer: The synthesized answer
                - sources: The source documents used
                - metadata: Pipeline execution metadata
        """
        start_time = time.time()
        self._total_queries += 1

        # Check cache if enabled
        cache_key = None
        if self._cache is not None:
            cache_key = self._get_cache_key(query, synthesis_prompt)
            if cache_key in self._cache:
                self._cache_stats["hits"] += 1
                cached_result = self._cache[cache_key].copy()
                cached_result["metadata"]["cache_hit"] = True
                cached_result["metadata"]["total_time"] = time.time() - start_time
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
            else:
                self._cache_stats["misses"] += 1

        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(query)
            retrieval_time = time.time() - retrieval_start

            logger.info(
                f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s"
            )

            # Step 2: Rerank documents
            rerank_start = time.time()
            if retrieved_docs:
                ranked_docs = self.ranker.rank_documents(query, retrieved_docs)
            else:
                ranked_docs = []
            rerank_time = time.time() - rerank_start

            logger.info(
                f"Reranked to {len(ranked_docs)} documents in {rerank_time:.2f}s"
            )

            # Step 3: Synthesize answer
            synthesis_start = time.time()
            synthesis_result = await self.synthesizer.synthesize_answer(
                query=query, documents=ranked_docs, custom_prompt=synthesis_prompt
            )
            synthesis_time = time.time() - synthesis_start

            # Prepare result
            total_time = time.time() - start_time

            result = {
                "answer": synthesis_result["answer"],
                "sources": synthesis_result["sources"],
                "metadata": {
                    "query_type": "fact_qa",
                    "retrieval_count": len(retrieved_docs),
                    "reranked_count": len(ranked_docs),
                    "retrieval_time": retrieval_time,
                    "rerank_time": rerank_time,
                    "synthesis_time": synthesis_time,
                    "total_time": total_time,
                    "language": self.config.answer_language,
                    "model": synthesis_result["metadata"]["model"],
                    "token_usage": synthesis_result["metadata"]["token_usage"],
                    "cache_hit": False,
                },
            }

            # Cache result if enabled
            if self._cache is not None and cache_key:
                # Limit cache size
                if len(self._cache) >= self.config.cache_size:
                    # Remove oldest entry (simple FIFO for now)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                self._cache[cache_key] = result.copy()

            # Log performance
            logger.info(
                f"Fact-based Q&A pipeline completed in {total_time:.2f}s - "
                f"retrieval: {retrieval_time:.2f}s, "
                f"rerank: {rerank_time:.2f}s, "
                f"synthesis: {synthesis_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in fact-based Q&A pipeline: {e}")
            raise

    def _get_cache_key(self, query: str, synthesis_prompt: str | None) -> str:
        """Generate a cache key for the query.

        Args:
            query: The query string
            synthesis_prompt: Optional custom prompt

        Returns:
            Cache key string
        """
        key_data = {
            "query": query,
            "synthesis_prompt": synthesis_prompt,
            "config": {
                "retriever_top_k": self.config.retriever_top_k,
                "reranker_top_k": self.config.reranker_top_k,
                "relevance_threshold": self.config.relevance_threshold,
                "answer_language": self.config.answer_language,
                "include_citations": self.config.include_citations,
            },
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def connect(self) -> None:
        """Connect pipeline components."""
        # Connect vector storage directly since retriever doesn't have connect method
        self.retriever.vector_storage.connect()
        logger.info("Fact-based Q&A pipeline connected")

    async def disconnect(self) -> None:
        """Disconnect pipeline components."""
        # Disconnect vector storage directly
        # (retriever doesn't have disconnect method)
        self.retriever.vector_storage.disconnect()
        logger.info("Fact-based Q&A pipeline disconnected")

    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline usage statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_queries": self._total_queries,
            "cache_stats": (
                self._cache_stats
                if self._cache is not None
                else {"cache_enabled": False}
            ),
            "component_stats": {
                "retriever": self.retriever.get_statistics(),
                "ranker": self.ranker.get_statistics(),
                "synthesizer": self.synthesizer.get_statistics(),
            },
        }

        if self._cache is not None and self._cache_stats["misses"] > 0:
            hit_rate = self._cache_stats["hits"] / (
                self._cache_stats["hits"] + self._cache_stats["misses"]
            )
            stats["cache_stats"]["hit_rate"] = hit_rate
            stats["cache_stats"]["cache_size"] = len(self._cache)

        return stats

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self._cache is not None:
            self._cache.clear()
            self._cache_stats = {"hits": 0, "misses": 0}
            logger.info("Fact-based Q&A pipeline cache cleared")
