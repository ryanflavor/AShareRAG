"""Unit tests for Ranker component."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.adapters.reranker_adapter import Qwen3RerankerAdapter, RerankResult
from src.components.ranker import Ranker, RankerConfig


class TestRanker:
    """Test cases for Ranker component."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return RankerConfig(relevance_threshold=0.5, batch_size=4, top_k=10)

    @pytest.fixture
    def mock_adapter(self):
        """Create mock reranker adapter."""
        adapter = Mock(spec=Qwen3RerankerAdapter)
        adapter.rerank = MagicMock()
        adapter.get_stats = MagicMock(
            return_value={"total_documents_processed": 0, "average_throughput": 0}
        )
        return adapter

    @pytest.fixture
    def ranker(self, config, mock_adapter):
        """Create ranker instance with mocks."""
        with patch(
            "src.components.ranker.Qwen3RerankerAdapter", return_value=mock_adapter
        ):
            return Ranker(config)

    def test_ranker_initialization(self, config):
        """Test ranker initializes correctly."""
        with patch("src.components.ranker.Qwen3RerankerAdapter") as mock_adapter_class:
            ranker = Ranker(config)

            assert ranker is not None
            assert ranker.config == config
            mock_adapter_class.assert_called_once()

    def test_rank_documents(self, ranker, mock_adapter):
        """Test basic document ranking."""
        query = "贵州茅台的主营业务"
        documents = [
            {"content": "贵州茅台主要从事白酒生产与销售。", "company_name": "贵州茅台"},
            {"content": "今天天气很好。", "company_name": "其他"},
            {"content": "贵州茅台是白酒行业龙头。", "company_name": "贵州茅台"},
        ]

        # Mock adapter response
        mock_results = [
            RerankResult("贵州茅台主要从事白酒生产与销售。", 0.9, 0),
            RerankResult("贵州茅台是白酒行业龙头。", 0.8, 2),
            RerankResult("今天天气很好。", 0.1, 1),  # Below threshold, will be filtered
        ]
        mock_adapter.rerank.return_value = mock_results

        results = ranker.rank_documents(query, documents)

        # Only 2 results above threshold (0.5)
        assert len(results) == 2
        assert results[0]["content"] == "贵州茅台主要从事白酒生产与销售。"
        assert results[0]["rerank_score"] == 0.9
        assert results[1]["rerank_score"] == 0.8

    def test_rank_with_relevance_threshold(self, ranker, mock_adapter):
        """Test ranking with relevance threshold filtering."""
        query = "test query"
        documents = [
            {"content": "relevant doc 1"},
            {"content": "irrelevant doc"},
            {"content": "relevant doc 2"},
        ]

        # Mock results with mixed scores
        mock_results = [
            RerankResult("relevant doc 1", 0.8, 0),
            RerankResult("relevant doc 2", 0.6, 2),
            RerankResult("irrelevant doc", 0.3, 1),  # Below threshold
        ]
        mock_adapter.rerank.return_value = mock_results

        results = ranker.rank_documents(query, documents)

        # Should filter out document with score < 0.5
        assert len(results) == 2
        assert all(doc["rerank_score"] >= 0.5 for doc in results)

    def test_rank_empty_documents(self, ranker, mock_adapter):
        """Test ranking with empty document list."""
        query = "test query"
        documents = []

        mock_adapter.rerank.return_value = []

        results = ranker.rank_documents(query, documents)

        assert results == []

    def test_rank_with_custom_top_k(self, ranker, mock_adapter):
        """Test ranking with custom top_k."""
        query = "test query"
        documents = [{"content": f"doc {i}"} for i in range(10)]

        # Mock 10 results
        mock_results = [RerankResult(f"doc {i}", 0.9 - i * 0.08, i) for i in range(10)]
        mock_adapter.rerank.return_value = mock_results[:5]  # Adapter returns top 5

        results = ranker.rank_documents(query, documents, top_k=5)

        assert len(results) == 5
        # Check that adapter was called with keyword arguments
        mock_adapter.rerank.assert_called_once()
        call_args = mock_adapter.rerank.call_args
        assert call_args.kwargs["query"] == query
        assert call_args.kwargs["documents"] == [f"doc {i}" for i in range(10)]
        assert call_args.kwargs["batch_size"] == ranker.config.batch_size
        assert call_args.kwargs["top_k"] == 5

    def test_rank_preserves_metadata(self, ranker, mock_adapter):
        """Test ranking preserves document metadata."""
        query = "test query"
        documents = [
            {
                "content": "doc 1",
                "company_name": "公司1",
                "metadata": {"source": "report", "year": 2023},
            },
            {
                "content": "doc 2",
                "company_name": "公司2",
                "metadata": {"source": "news", "date": "2024-01-01"},
            },
        ]

        mock_results = [RerankResult("doc 2", 0.9, 1), RerankResult("doc 1", 0.7, 0)]
        mock_adapter.rerank.return_value = mock_results

        results = ranker.rank_documents(query, documents)

        assert len(results) == 2
        # First result should be doc 2 with its metadata
        assert results[0]["content"] == "doc 2"
        assert results[0]["company_name"] == "公司2"
        assert results[0]["metadata"]["source"] == "news"
        assert results[0]["rerank_score"] == 0.9
        assert results[0]["rerank_rank"] == 1

    def test_rank_handles_missing_content_field(self, ranker, mock_adapter):
        """Test ranking handles documents without content field."""
        query = "test query"
        documents = [
            {"text": "doc 1", "company_name": "公司1"},  # Different field name
            {"content": "doc 2", "company_name": "公司2"},
        ]

        # Should handle gracefully
        with pytest.raises(KeyError):
            ranker.rank_documents(query, documents)

    def test_rank_with_custom_text_field(self, ranker, mock_adapter):
        """Test ranking with custom text field."""
        query = "test query"
        documents = [{"text": "doc 1", "id": 1}, {"text": "doc 2", "id": 2}]

        mock_results = [RerankResult("doc 2", 0.8, 1), RerankResult("doc 1", 0.6, 0)]
        mock_adapter.rerank.return_value = mock_results

        results = ranker.rank_documents(query, documents, text_field="text")

        assert len(results) == 2
        assert results[0]["text"] == "doc 2"
        assert results[0]["id"] == 2

    def test_get_statistics(self, ranker, mock_adapter):
        """Test getting ranker statistics."""
        mock_adapter.get_stats.return_value = {
            "total_documents_processed": 100,
            "average_throughput": 10.5,
            "model_name": "Qwen/Qwen3-Reranker-4B",
        }

        stats = ranker.get_statistics()

        assert "reranker_stats" in stats
        assert stats["reranker_stats"]["total_documents_processed"] == 100
        assert (
            stats["config"]["relevance_threshold"] == ranker.config.relevance_threshold
        )

    def test_rank_logs_performance(self, ranker, mock_adapter):
        """Test ranking logs performance metrics."""
        with patch("src.components.ranker.logger") as mock_logger:
            query = "test query"
            documents = [{"content": "doc 1"}]

            mock_adapter.rerank.return_value = [RerankResult("doc 1", 0.8, 0)]

            ranker.rank_documents(query, documents)

            # Check performance logging
            assert mock_logger.info.called
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Ranking completed" in call for call in log_calls)

    def test_rank_handles_adapter_errors(self, ranker, mock_adapter):
        """Test ranking handles adapter errors gracefully."""
        query = "test query"
        documents = [{"content": "doc 1"}]

        mock_adapter.rerank.side_effect = Exception("Model error")

        with pytest.raises(Exception) as exc_info:
            ranker.rank_documents(query, documents)

        assert "Model error" in str(exc_info.value)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid relevance threshold
        with pytest.raises(ValueError):
            RankerConfig(relevance_threshold=1.5)  # Should be between 0 and 1

        with pytest.raises(ValueError):
            RankerConfig(relevance_threshold=-0.1)

        # Invalid batch size
        with pytest.raises(ValueError):
            RankerConfig(batch_size=0)

        # Invalid top_k
        with pytest.raises(ValueError):
            RankerConfig(top_k=-1)
