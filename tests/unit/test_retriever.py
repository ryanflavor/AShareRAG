"""Unit tests for VectorRetriever component."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.components.embedding_service import EmbeddingService
from src.components.retriever import VectorRetriever
from src.components.vector_storage import VectorStorage


class TestVectorRetriever:
    """Test cases for VectorRetriever component."""

    @pytest.fixture
    def mock_vector_storage(self):
        """Create mock vector storage."""
        mock = Mock(spec=VectorStorage)
        mock.search = MagicMock()
        return mock

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock = Mock(spec=EmbeddingService)
        mock.generate_embeddings = MagicMock()
        return mock

    @pytest.fixture
    def retriever(self, mock_vector_storage, mock_embedding_service):
        """Create retriever instance with mocks."""
        return VectorRetriever(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            top_k=10,
        )

    def test_retriever_initialization(self, retriever):
        """Test retriever initializes correctly."""
        assert retriever is not None
        assert retriever.top_k == 10

    def test_retrieve_with_query(
        self, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test basic retrieval with query."""
        # Arrange
        query = "贵州茅台的主营业务是什么?"
        query_embedding = np.random.rand(1536).tolist()

        mock_embedding_service.generate_embeddings.return_value = [query_embedding]
        mock_vector_storage.search.return_value = [
            {
                "content": "贵州茅台主要从事白酒生产与销售。",
                "company_name": "贵州茅台",
                "score": 0.95,
                "metadata": {"source": "annual_report_2023"},
            }
        ]

        # Act
        results = retriever.retrieve(query)

        # Assert
        assert len(results) == 1
        assert results[0]["content"] == "贵州茅台主要从事白酒生产与销售。"
        assert results[0]["company_name"] == "贵州茅台"
        assert results[0]["score"] == 0.95
        mock_embedding_service.generate_embeddings.assert_called_once_with([query])
        mock_vector_storage.search.assert_called_once()

    def test_retrieve_with_company_filter(
        self, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test retrieval with company entity extraction and filtering."""
        # Arrange
        query = "比亚迪的新能源汽车销量如何?"
        query_embedding = np.random.rand(1536).tolist()

        mock_embedding_service.generate_embeddings.return_value = [query_embedding]
        mock_vector_storage.search.return_value = [
            {
                "content": "比亚迪2023年新能源汽车销量突破300万辆。",
                "company_name": "比亚迪",
                "score": 0.92,
                "metadata": {"source": "annual_report_2023"},
            }
        ]

        # Act
        results = retriever.retrieve(query, company_filter="比亚迪")

        # Assert
        assert len(results) == 1
        assert results[0]["company_name"] == "比亚迪"
        # Verify search was called with company filter
        call_args = mock_vector_storage.search.call_args
        assert call_args[1].get("filter_company") == "比亚迪"

    def test_auto_extract_company_from_query(self, retriever):
        """Test automatic company extraction from query."""
        # Act
        company = retriever.extract_company_from_query("贵州茅台的财务状况如何?")

        # Assert
        assert company == "贵州茅台"

    def test_extract_company_handles_multiple_formats(self, retriever):
        """Test company extraction handles various query formats."""
        test_cases = [
            ("贵州茅台的主营业务", "贵州茅台"),
            ("请问比亚迪的新能源技术", "比亚迪"),
            ("宁德时代在电池领域的地位", "宁德时代"),
            ("分析一下中国平安的保险业务", "中国平安"),
            ("没有公司名称的查询", None),
        ]

        for query, expected_company in test_cases:
            company = retriever.extract_company_from_query(query)
            assert company == expected_company, f"Failed for query: {query}"

    def test_retrieve_with_empty_results(
        self, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test retrieval when no results found."""
        # Arrange
        query = "不存在的公司信息"
        mock_embedding_service.generate_embeddings.return_value = [
            np.random.rand(1536).tolist()
        ]
        mock_vector_storage.search.return_value = []

        # Act
        results = retriever.retrieve(query)

        # Assert
        assert results == []

    def test_retrieve_with_top_k_limit(
        self, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test retrieval respects top_k limit."""
        # Arrange
        query = "A股上市公司"
        mock_embedding_service.generate_embeddings.return_value = [
            np.random.rand(1536).tolist()
        ]

        # Create 20 mock results
        mock_results = [
            {
                "content": f"公司{i}的信息",
                "company_name": f"公司{i}",
                "score": 0.9 - i * 0.01,
                "metadata": {"source": f"report_{i}"},
            }
            for i in range(20)
        ]
        mock_vector_storage.search.return_value = mock_results

        # Act
        results = retriever.retrieve(query)

        # Assert
        assert len(results) == 10  # Should respect top_k=10
        assert results[0]["score"] > results[9]["score"]  # Should be sorted by score

    def test_retrieve_handles_embedding_error(
        self, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test retrieval handles embedding generation errors gracefully."""
        # Arrange
        query = "测试查询"
        mock_embedding_service.generate_embeddings.side_effect = Exception(
            "Embedding model error"
        )

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            retriever.retrieve(query)
        assert "Embedding model error" in str(exc_info.value)

    def test_retrieve_handles_vector_storage_error(
        self, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test retrieval handles vector storage errors gracefully."""
        # Arrange
        query = "测试查询"
        mock_embedding_service.generate_embeddings.return_value = [
            np.random.rand(1536).tolist()
        ]
        mock_vector_storage.search.side_effect = Exception("Vector storage error")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            retriever.retrieve(query)
        assert "Vector storage error" in str(exc_info.value)

    def test_retrieve_with_custom_top_k(
        self, mock_vector_storage, mock_embedding_service
    ):
        """Test retriever with custom top_k value."""
        # Arrange
        custom_retriever = VectorRetriever(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            top_k=5,
        )

        # Assert
        assert custom_retriever.top_k == 5

    @patch("src.components.retriever.logger")
    def test_retrieve_logs_performance(
        self, mock_logger, retriever, mock_embedding_service, mock_vector_storage
    ):
        """Test retrieval logs performance metrics."""
        # Arrange
        query = "测试查询"
        mock_embedding_service.generate_embeddings.return_value = [
            np.random.rand(1536).tolist()
        ]
        mock_vector_storage.search.return_value = []

        # Act
        retriever.retrieve(query)

        # Assert
        # Check that performance metrics were logged
        assert mock_logger.info.called
        log_calls = mock_logger.info.call_args_list
        assert any("Embedding generation time" in str(call) for call in log_calls)
        assert any("Vector search time" in str(call) for call in log_calls)
