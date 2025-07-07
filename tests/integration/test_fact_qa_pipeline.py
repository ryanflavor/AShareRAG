"""Integration tests for the fact-based Q&A pipeline.

This module tests the end-to-end functionality of the fact-based Q&A pipeline,
integrating retrieval, reranking, and answer synthesis components.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.adapters.llm_adapter import LLMResponse
from src.adapters.reranker_adapter import RerankResult
from src.components.answer_synthesizer import AnswerSynthesizer
from src.components.ranker import Ranker
from src.components.retriever import VectorRetriever
from src.pipeline.fact_qa_pipeline import FactQAPipeline, FactQAPipelineConfig


class TestFactQAPipelineIntegration:
    """Integration tests for the fact-based Q&A pipeline."""

    @pytest.fixture
    def mock_vector_storage(self):
        """Create a mock vector storage."""
        storage = Mock()
        storage.connect = Mock()
        storage.disconnect = Mock()
        storage.search = Mock(return_value=[])
        return storage

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.generate_embeddings = Mock(return_value=[[0.1] * 768])
        return service

    @pytest.fixture
    def mock_llm_adapter(self):
        """Create a mock LLM adapter."""
        adapter = Mock()
        adapter.generate_async = AsyncMock()
        return adapter

    @pytest.fixture
    def pipeline_config(self):
        """Create a test pipeline configuration."""
        return FactQAPipelineConfig(
            retriever_top_k=10,
            reranker_top_k=5,
            relevance_threshold=0.5,
            answer_max_tokens=500,
            answer_temperature=0.7,
            answer_language="Chinese",
            include_citations=True,
            enable_caching=False,
        )

    @pytest.fixture
    def pipeline(
        self,
        pipeline_config,
        mock_vector_storage,
        mock_embedding_service,
        mock_llm_adapter,
    ):
        """Create a FactQAPipeline instance for testing."""
        # Mock the Ranker's internal reranker to control behavior
        with patch("src.components.ranker.Qwen3RerankerAdapter") as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker.rerank = Mock(return_value=[])
            mock_reranker_class.return_value = mock_reranker

            pipeline = FactQAPipeline(
                config=pipeline_config,
                vector_storage=mock_vector_storage,
                embedding_service=mock_embedding_service,
                llm_adapter=mock_llm_adapter,
            )
            # Store mock for tests to use
            pipeline._test_mock_reranker = mock_reranker
            return pipeline

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline.config is not None
        assert isinstance(pipeline.retriever, VectorRetriever)
        assert isinstance(pipeline.ranker, Ranker)
        assert isinstance(pipeline.synthesizer, AnswerSynthesizer)
        assert pipeline._cache is None  # Caching disabled in config

    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(
        self, pipeline, mock_vector_storage, mock_llm_adapter
    ):
        """Test end-to-end query processing."""
        # Setup mock responses
        mock_vector_storage.search.return_value = [
            {
                "content": "比亚迪2023年营收达到6023亿元，同比增长42.04%。",
                "metadata": {
                    "company_name": "比亚迪",
                    "source": "annual_report_2023",
                    "page": 15,
                },
            },
            {
                "content": "比亚迪新能源汽车销量创历史新高。",
                "metadata": {
                    "company_name": "比亚迪",
                    "source": "news_2023",
                    "page": 1,
                },
            },
        ]

        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(
                document="比亚迪2023年营收达到6023亿元，同比增长42.04%。",
                score=0.92,
                original_index=0,
                metadata={
                    "company_name": "比亚迪",
                    "source": "annual_report_2023",
                    "page": 15,
                },
            )
        ]

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="根据2023年财报，比亚迪营收达到6023亿元，同比增长42.04%[1]。",
            model="deepseek-v3",
            usage={"prompt_tokens": 300, "completion_tokens": 50},
        )

        # Execute pipeline
        result = await pipeline.process_query("比亚迪2023年营收是多少？")

        # Verify result structure
        assert "answer" in result
        assert "sources" in result
        assert "metadata" in result

        # Verify answer content
        assert "6023亿元" in result["answer"]
        assert "[1]" in result["answer"]  # Citation

        # Verify sources
        assert len(result["sources"]) == 1
        assert result["sources"][0]["rerank_score"] == 0.92

        # Verify metadata
        assert result["metadata"]["retrieval_count"] == 2
        assert result["metadata"]["reranked_count"] == 1
        assert result["metadata"]["query_type"] == "fact_qa"
        assert "total_time" in result["metadata"]

    @pytest.mark.asyncio
    async def test_pipeline_with_no_results(
        self, pipeline, mock_vector_storage, mock_llm_adapter
    ):
        """Test pipeline behavior when no documents are found."""
        # Setup empty results
        mock_vector_storage.search.return_value = []

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="抱歉，未找到与您查询相关的信息。",
            model="deepseek-v3",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )

        # Execute pipeline
        result = await pipeline.process_query("不存在的公司信息")

        # Verify handling of no results
        assert result["answer"] == "抱歉，未找到与您查询相关的信息。"
        assert result["sources"] == []
        assert result["metadata"]["retrieval_count"] == 0
        assert result["metadata"]["reranked_count"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_with_reranking_filter(
        self, pipeline, mock_vector_storage, mock_llm_adapter
    ):
        """Test that low-relevance documents are filtered out."""
        # Setup mock responses
        mock_vector_storage.search.return_value = [
            {"content": "相关文档", "metadata": {"company_name": "公司A"}},
            {"content": "不太相关的文档", "metadata": {"company_name": "公司B"}},
        ]

        # Only high-score document passes reranking
        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(
                document="相关文档",
                score=0.85,
                original_index=0,
                metadata={"company_name": "公司A"},
            ),
            RerankResult(
                document="不太相关的文档",
                score=0.3,  # Below threshold
                original_index=1,
                metadata={"company_name": "公司B"},
            ),
        ]

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="基于相关文档的答案。",
            model="deepseek-v3",
            usage={"prompt_tokens": 200, "completion_tokens": 30},
        )

        # Execute pipeline
        result = await pipeline.process_query("查询测试")

        # Verify only high-relevance document is used
        assert result["metadata"]["retrieval_count"] == 2
        assert result["metadata"]["reranked_count"] == 1  # Only 1 passes threshold
        assert len(result["sources"]) == 1
        assert result["sources"][0]["rerank_score"] == 0.85

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline, mock_vector_storage):
        """Test pipeline error handling."""
        # Setup retrieval failure
        mock_vector_storage.search.side_effect = Exception("Vector storage error")

        # Execute pipeline and expect error
        with pytest.raises(Exception) as exc_info:
            await pipeline.process_query("测试查询")

        assert "Vector storage error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pipeline_with_caching_enabled(
        self, mock_vector_storage, mock_embedding_service, mock_llm_adapter
    ):
        """Test pipeline with caching enabled."""
        # Create config with caching enabled
        config = FactQAPipelineConfig(enable_caching=True)

        with patch("src.components.ranker.Qwen3RerankerAdapter") as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker.rerank = Mock(return_value=[])
            mock_reranker_class.return_value = mock_reranker

            pipeline = FactQAPipeline(
                config=config,
                vector_storage=mock_vector_storage,
                embedding_service=mock_embedding_service,
                llm_adapter=mock_llm_adapter,
            )
            pipeline._test_mock_reranker = mock_reranker

        # Setup mock responses
        mock_vector_storage.search.return_value = [{"content": "test", "metadata": {}}]
        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(document="test", score=0.9, original_index=0, metadata={})
        ]
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="Cached answer",
            model="deepseek-v3",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )

        # First query
        result1 = await pipeline.process_query("测试查询")
        assert not result1["metadata"].get("cache_hit", False)

        # Second identical query should hit cache
        result2 = await pipeline.process_query("测试查询")
        assert result2["metadata"]["cache_hit"] is True
        assert result2["answer"] == result1["answer"]

        # Verify vector search was only called once
        assert mock_vector_storage.search.call_count == 1

    @pytest.mark.asyncio
    async def test_pipeline_performance_logging(
        self, pipeline, mock_vector_storage, mock_llm_adapter, caplog
    ):
        """Test that pipeline logs performance metrics."""
        # Setup mock responses
        mock_vector_storage.search.return_value = [{"content": "test", "metadata": {}}]
        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(document="test", score=0.9, original_index=0, metadata={})
        ]
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="Test answer",
            model="deepseek-v3",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )

        # Execute with logging
        with caplog.at_level("INFO"):
            result = await pipeline.process_query("测试查询")

        # Verify performance logging
        assert any(
            "Fact-based Q&A pipeline completed" in record.message
            for record in caplog.records
        )
        assert "total_time" in result["metadata"]
        assert result["metadata"]["total_time"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_requests(
        self, pipeline, mock_vector_storage, mock_llm_adapter
    ):
        """Test handling of concurrent pipeline requests."""
        # Setup mock responses
        mock_vector_storage.search.return_value = [{"content": "test", "metadata": {}}]
        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(document="test", score=0.9, original_index=0, metadata={})
        ]

        # Different answers for different queries
        async def generate_answer(prompt, **kwargs):
            if "查询1" in prompt:
                return LLMResponse(
                    content="答案1",
                    model="deepseek-v3",
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )
            else:
                return LLMResponse(
                    content="答案2",
                    model="deepseek-v3",
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                )

        mock_llm_adapter.generate_async.side_effect = generate_answer

        # Execute concurrent queries
        tasks = [
            pipeline.process_query("查询1"),
            pipeline.process_query("查询2"),
            pipeline.process_query("查询1"),  # Duplicate
        ]

        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 3
        assert results[0]["answer"] == "答案1"
        assert results[1]["answer"] == "答案2"
        assert results[2]["answer"] == "答案1"

    @pytest.mark.asyncio
    async def test_pipeline_with_english_mode(
        self, mock_vector_storage, mock_embedding_service, mock_llm_adapter
    ):
        """Test pipeline in English mode."""
        # Create config for English
        config = FactQAPipelineConfig(answer_language="English")

        with patch("src.components.ranker.Qwen3RerankerAdapter") as mock_reranker_class:
            mock_reranker = Mock()
            mock_reranker.rerank = Mock(return_value=[])
            mock_reranker_class.return_value = mock_reranker

            pipeline = FactQAPipeline(
                config=config,
                vector_storage=mock_vector_storage,
                embedding_service=mock_embedding_service,
                llm_adapter=mock_llm_adapter,
            )
            pipeline._test_mock_reranker = mock_reranker

        # Setup mock responses
        mock_vector_storage.search.return_value = [
            {
                "content": "BYD revenue reached 602.3 billion RMB in 2023.",
                "metadata": {"company_name": "BYD", "source": "report"},
            }
        ]

        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(
                document="BYD revenue reached 602.3 billion RMB in 2023.",
                score=0.95,
                original_index=0,
                metadata={"company_name": "BYD", "source": "report"},
            )
        ]

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="BYD's revenue was 602.3 billion RMB in 2023[1].",
            model="deepseek-v3",
            usage={"prompt_tokens": 200, "completion_tokens": 30},
        )

        # Execute pipeline
        result = await pipeline.process_query("What was BYD's revenue in 2023?")

        # Verify English response
        assert "602.3 billion" in result["answer"]
        assert result["metadata"]["language"] == "English"

    def test_pipeline_statistics(self, pipeline):
        """Test statistics collection."""
        stats = pipeline.get_statistics()

        assert "total_queries" in stats
        assert "cache_stats" in stats
        assert "component_stats" in stats
        assert stats["total_queries"] == 0  # No queries yet

    @pytest.mark.asyncio
    async def test_pipeline_lifecycle_management(self, pipeline, mock_vector_storage):
        """Test pipeline lifecycle methods."""
        # Test connection
        await pipeline.connect()
        mock_vector_storage.connect.assert_called_once()

        # Test disconnection
        await pipeline.disconnect()
        mock_vector_storage.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_prompt(
        self, pipeline, mock_vector_storage, mock_llm_adapter
    ):
        """Test pipeline with custom synthesis prompt."""
        # Setup mock responses
        mock_vector_storage.search.return_value = [
            {"content": "test doc", "metadata": {}}
        ]
        pipeline._test_mock_reranker.rerank.return_value = [
            RerankResult(document="test doc", score=0.9, original_index=0, metadata={})
        ]
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="Custom prompt answer",
            model="deepseek-v3",
            usage={"prompt_tokens": 150, "completion_tokens": 25},
        )

        # Execute with custom prompt
        custom_prompt = "Based on: {documents}\n\nAnswer this: {query}"
        result = await pipeline.process_query(
            "Test query", synthesis_prompt=custom_prompt
        )

        # Verify custom prompt was used
        assert result["answer"] == "Custom prompt answer"
        assert mock_llm_adapter.generate_async.called
