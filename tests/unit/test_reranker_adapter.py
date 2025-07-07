"""Unit tests for Qwen3RerankerAdapter."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.adapters.reranker_adapter import Qwen3RerankerAdapter, RerankerConfig


class TestQwen3RerankerAdapter:
    """Test cases for Qwen3RerankerAdapter."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return RerankerConfig(
            model_name="Qwen/Qwen3-Reranker-4B",
            device="cpu",
            dtype=torch.float32,
            batch_size=4,
            max_length=512,
            use_bf16=False,
            cache_dir=None,
        )

    @pytest.fixture
    def adapter(self, config):
        """Create adapter instance."""
        with patch("src.adapters.reranker_adapter.AutoTokenizer"):
            with patch("src.adapters.reranker_adapter.AutoModelForCausalLM"):
                return Qwen3RerankerAdapter(config)

    def test_adapter_initialization(self, config):
        """Test adapter initializes correctly."""
        with patch("src.adapters.reranker_adapter.AutoTokenizer") as mock_tokenizer:
            with patch(
                "src.adapters.reranker_adapter.AutoModelForCausalLM"
            ) as mock_model:
                # Configure mocks
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                adapter = Qwen3RerankerAdapter(config)

                assert adapter is not None
                assert adapter.config == config
                mock_tokenizer.from_pretrained.assert_called_once()
                mock_model.from_pretrained.assert_called_once()

    def test_format_input(self, adapter):
        """Test input formatting."""
        query = "贵州茅台的主营业务是什么?"
        document = "贵州茅台主要从事白酒生产与销售。"

        formatted = adapter._format_input(query, document)

        assert query in formatted
        assert document in formatted
        assert "<|im_start|>" in formatted
        assert "<|im_end|>" in formatted

    def test_rerank_single_document(self, adapter):
        """Test reranking with single document."""
        query = "贵州茅台的主营业务"
        documents = ["贵州茅台主要从事白酒生产与销售。"]

        # Mock the model output
        mock_logits = torch.tensor([[[0.1, 0.9, 0.2]]])  # Dummy logits
        adapter.model = MagicMock()
        adapter.model.return_value = MagicMock(logits=mock_logits)

        # Mock token IDs
        adapter.token_yes = 1
        adapter.token_no = 0

        results = adapter.rerank(query, documents)

        assert len(results) == 1
        assert 0 <= results[0].score <= 1
        assert results[0].document == documents[0]
        assert results[0].original_index == 0

    def test_rerank_multiple_documents(self, adapter):
        """Test reranking with multiple documents."""
        query = "贵州茅台的财务状况"
        documents = [
            "贵州茅台2023年营收超过1000亿元。",
            "今天天气很好。",
            "贵州茅台的净利润持续增长。",
            "Python是一种编程语言。",
        ]

        # Mock different scores for each document
        mock_scores = [0.9, 0.1, 0.8, 0.05]
        adapter._process_batch = MagicMock(return_value=mock_scores)

        results = adapter.rerank(query, documents)

        assert len(results) == 4
        # Should be sorted by score descending
        assert results[0].score > results[1].score
        assert results[1].score > results[2].score
        assert results[0].document == documents[0]  # Highest score
        assert results[-1].document == documents[3]  # Lowest score

    def test_rerank_with_top_k(self, adapter):
        """Test reranking with top_k parameter."""
        query = "test query"
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        mock_scores = [0.5, 0.7, 0.3, 0.9, 0.1]
        adapter._process_batch = MagicMock(return_value=mock_scores)

        results = adapter.rerank(query, documents, top_k=3)

        assert len(results) == 3
        # Check top 3 scores
        assert results[0].score == 0.9
        assert results[1].score == 0.7
        assert results[2].score == 0.5

    def test_rerank_empty_documents(self, adapter):
        """Test reranking with empty document list."""
        query = "test query"
        documents = []

        results = adapter.rerank(query, documents)

        assert results == []

    def test_batch_processing(self, adapter):
        """Test batch processing works correctly."""
        query = "test query"
        # Create more documents than batch size
        documents = [f"document {i}" for i in range(10)]
        adapter.config.batch_size = 3

        # Mock scores for each batch
        def mock_process_batch(q, docs):
            return [0.5] * len(docs)

        adapter._process_batch = MagicMock(side_effect=mock_process_batch)

        results = adapter.rerank(query, documents)

        assert len(results) == 10
        # Should have called process_batch 4 times (3+3+3+1)
        assert adapter._process_batch.call_count == 4

    def test_rerank_with_metadata(self, adapter):
        """Test reranking documents with metadata."""
        query = "贵州茅台"
        documents = [
            {"text": "贵州茅台是白酒龙头", "id": 1, "source": "report"},
            {"text": "今日股市行情", "id": 2, "source": "news"},
            {"text": "茅台酒历史悠久", "id": 3, "source": "wiki"},
        ]

        mock_scores = [0.9, 0.2, 0.7]
        adapter._process_batch = MagicMock(return_value=mock_scores)

        results = adapter.rerank_with_metadata(query, documents)

        assert len(results) == 3
        # Should be sorted by score
        assert results[0]["id"] == 1
        assert results[0]["rerank_score"] == 0.9
        assert results[0]["rerank_rank"] == 1
        assert results[1]["id"] == 3
        assert results[2]["id"] == 2

    @patch("torch.cuda.is_available")
    def test_cuda_device_handling(self, mock_cuda, config):
        """Test CUDA device handling."""
        mock_cuda.return_value = True
        config.device = "cuda"

        with patch("src.adapters.reranker_adapter.AutoTokenizer"):
            with patch(
                "src.adapters.reranker_adapter.AutoModelForCausalLM"
            ) as mock_model:
                mock_model_instance = MagicMock()
                mock_model.from_pretrained.return_value = mock_model_instance

                Qwen3RerankerAdapter(config)

                # Should use device_map="auto" for CUDA
                mock_model.from_pretrained.assert_called_with(
                    config.model_name,
                    torch_dtype=config.dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=None,
                )

    def test_error_handling_in_process_batch(self, adapter):
        """Test error handling during batch processing."""
        query = "test query"
        documents = ["doc1", "doc2"]

        # Simulate an error during processing
        adapter._process_batch = MagicMock(side_effect=Exception("Processing error"))

        with pytest.raises(Exception) as exc_info:
            adapter.rerank(query, documents)

        assert "Processing error" in str(exc_info.value)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    def test_memory_efficient_mode(self, mock_mem_allocated, mock_cuda, adapter):
        """Test memory efficient mode with OOM recovery."""
        mock_cuda.return_value = True
        mock_mem_allocated.return_value = 1024 * 1024 * 1024  # 1GB

        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        # First call raises OOM, then succeeds
        oom_error = RuntimeError("CUDA out of memory")
        adapter._process_batch = MagicMock(side_effect=[oom_error, [0.5], [0.6], [0.7]])

        results = adapter.rerank(query, documents)

        assert len(results) == 3
        # Should have called process_batch 4 times (1 failed + 3 single doc)
        assert adapter._process_batch.call_count == 4

    def test_get_stats(self, adapter):
        """Test statistics retrieval."""
        # Process some documents first
        adapter.total_processed = 100
        adapter.total_time = 10.0

        stats = adapter.get_stats()

        assert stats["total_documents_processed"] == 100
        assert stats["total_processing_time"] == 10.0
        assert stats["average_throughput"] == 10.0
        assert stats["model_name"] == adapter.config.model_name
        assert stats["device"] == adapter.config.device
