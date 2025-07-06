"""Unit tests for Embedding Service component."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.components.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock Qwen3EmbeddingManager."""
        manager = Mock()
        manager.load_model.return_value = True
        manager.encode_texts.return_value = np.random.rand(5, 2560).astype(np.float32)
        manager.embedding_dim = 2560
        manager.model_name = "Qwen/Qwen3-Embedding-4B"
        return manager

    @pytest.fixture
    def embedding_service(self, mock_embedding_manager):
        """Create an EmbeddingService instance with mocked dependencies."""
        with patch(
            "src.components.embedding_service.Qwen3EmbeddingManager"
        ) as mock_class:
            mock_class.return_value = mock_embedding_manager
            service = EmbeddingService()
            return service

    def test_init(self):
        """Test EmbeddingService initialization."""
        with patch(
            "src.components.embedding_service.Qwen3EmbeddingManager"
        ) as mock_class:
            service = EmbeddingService(
                model_name="custom-model",
                device="cuda",
                embedding_dim=1024,
                batch_size=64,
            )

            # Verify initialization parameters
            assert service.model_name == "custom-model"
            assert service.device == "cuda"
            assert service.embedding_dim == 1024
            assert service.batch_size == 64

            # Verify manager was created with correct parameters
            mock_class.assert_called_once_with(
                model_name="custom-model", device="cuda", embedding_dim=1024
            )

    def test_load_model_success(self, embedding_service, mock_embedding_manager):
        """Test successful model loading."""
        result = embedding_service.load_model()

        assert result is True
        mock_embedding_manager.load_model.assert_called_once()

    def test_load_model_failure(self, embedding_service, mock_embedding_manager):
        """Test model loading failure."""
        mock_embedding_manager.load_model.return_value = False

        result = embedding_service.load_model()

        assert result is False

    def test_generate_embeddings_single_text(
        self, embedding_service, mock_embedding_manager
    ):
        """Test embedding generation for a single text."""
        texts = ["This is a test document about AI technology."]
        mock_embedding_manager.encode_texts.return_value = np.random.rand(
            1, 2560
        ).astype(np.float32)

        embeddings = embedding_service.generate_embeddings(texts)

        assert embeddings is not None
        assert embeddings.shape == (1, 2560)
        mock_embedding_manager.encode_texts.assert_called_once_with(
            texts, batch_size=32, show_progress=False
        )

    def test_generate_embeddings_batch(self, embedding_service, mock_embedding_manager):
        """Test embedding generation for batch of texts."""
        texts = [f"Document {i}" for i in range(100)]
        mock_embedding_manager.encode_texts.return_value = np.random.rand(
            100, 2560
        ).astype(np.float32)

        embeddings = embedding_service.generate_embeddings(texts)

        assert embeddings is not None
        assert embeddings.shape == (100, 2560)
        mock_embedding_manager.encode_texts.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress=True,  # Should show progress for large batches
        )

    def test_generate_embeddings_empty_list(self, embedding_service):
        """Test embedding generation with empty text list."""
        embeddings = embedding_service.generate_embeddings([])

        assert embeddings is not None
        assert embeddings.shape == (0, 2560)

    def test_generate_embeddings_model_not_loaded(
        self, embedding_service, mock_embedding_manager
    ):
        """Test embedding generation when model is not loaded."""
        mock_embedding_manager.model = None

        embeddings = embedding_service.generate_embeddings(["test"])

        assert embeddings is None

    def test_process_documents_with_metadata(
        self, embedding_service, mock_embedding_manager
    ):
        """Test processing documents with comprehensive metadata."""
        documents = [
            {
                "text": "Document about company A",
                "doc_id": "doc1",
                "chunk_index": 0,
                "company_name": "Company A",
                "entities": [
                    {"text": "Company A", "type": "COMPANY"},
                    {"text": "AI Product", "type": "PRODUCT"},
                ],
                "relations": [["Company A", "develops", "AI Product"]],
                "source_file": "corpus.json",
            },
            {
                "text": "Another document about technology",
                "doc_id": "doc2",
                "chunk_index": 0,
                "company_name": "Tech Corp",
                "entities": [
                    {"text": "Tech Corp", "type": "COMPANY"},
                    {"text": "Beijing", "type": "LOCATION"},
                ],
                "relations": [["Tech Corp", "located_in", "Beijing"]],
                "source_file": "corpus.json",
            },
        ]

        mock_embedding_manager.encode_texts.return_value = np.random.rand(
            2, 2560
        ).astype(np.float32)

        results = embedding_service.process_documents(documents)

        assert len(results) == 2

        # Check first document
        assert results[0]["id"] == "doc1_0"
        assert results[0]["text"] == "Document about company A"
        assert results[0]["vector"].shape == (2560,)
        assert results[0]["company_name"] == "Company A"
        assert len(results[0]["entities"]) == 2
        assert results[0]["entities"][0]["type"] == "COMPANY"
        assert results[0]["relations_count"] == 1

        # Check second document
        assert results[1]["id"] == "doc2_0"
        assert results[1]["entities"][1]["type"] == "LOCATION"

    def test_process_documents_with_missing_metadata(
        self, embedding_service, mock_embedding_manager
    ):
        """Test processing documents with missing optional metadata."""
        documents = [
            {
                "text": "Simple document",
                "doc_id": "doc1",
                "chunk_index": 0,
                "company_name": "Company X",
                # Missing entities and relations
            }
        ]

        mock_embedding_manager.encode_texts.return_value = np.random.rand(
            1, 2560
        ).astype(np.float32)

        results = embedding_service.process_documents(documents)

        assert len(results) == 1
        assert results[0]["entities"] == []
        assert results[0]["relations"] == []
        assert results[0]["relations_count"] == 0

    def test_process_documents_batch_processing(
        self, embedding_service, mock_embedding_manager
    ):
        """Test batch processing of large document sets."""
        # Create 150 documents to test batching
        documents = [
            {
                "text": f"Document {i}",
                "doc_id": f"doc{i}",
                "chunk_index": 0,
                "company_name": f"Company {i}",
                "entities": [],
                "relations": [],
            }
            for i in range(150)
        ]

        # Mock returns for batch processing
        mock_embedding_manager.encode_texts.side_effect = [
            np.random.rand(32, 2560).astype(np.float32),  # First batch
            np.random.rand(32, 2560).astype(np.float32),  # Second batch
            np.random.rand(32, 2560).astype(np.float32),  # Third batch
            np.random.rand(32, 2560).astype(np.float32),  # Fourth batch
            np.random.rand(22, 2560).astype(np.float32),  # Last batch
        ]

        results = embedding_service.process_documents(documents)

        assert len(results) == 150
        assert mock_embedding_manager.encode_texts.call_count == 5  # 5 batches

    def test_memory_optimization_features(
        self, embedding_service, mock_embedding_manager
    ):
        """Test memory optimization and GPU handling."""
        # Test that memory management features are properly utilized
        with patch("src.components.embedding_service.clear_memory") as mock_clear:
            documents = [
                {
                    "text": f"Doc {i}",
                    "doc_id": f"d{i}",
                    "chunk_index": 0,
                    "company_name": "Test",
                }
                for i in range(100)
            ]

            mock_embedding_manager.encode_texts.return_value = np.random.rand(
                100, 2560
            ).astype(np.float32)

            embedding_service.process_documents(documents)

            # Memory should be cleared after processing
            assert mock_clear.called

    def test_error_handling_embedding_failure(
        self, embedding_service, mock_embedding_manager
    ):
        """Test error handling when embedding generation fails."""
        mock_embedding_manager.encode_texts.return_value = None

        documents = [
            {"text": "Test", "doc_id": "d1", "chunk_index": 0, "company_name": "Test"}
        ]

        results = embedding_service.process_documents(documents)

        assert results == []

    def test_timestamp_generation(self, embedding_service, mock_embedding_manager):
        """Test that processing timestamps are generated correctly."""
        from datetime import datetime

        documents = [
            {"text": "Test", "doc_id": "d1", "chunk_index": 0, "company_name": "Test"}
        ]
        mock_embedding_manager.encode_texts.return_value = np.random.rand(
            1, 2560
        ).astype(np.float32)

        results = embedding_service.process_documents(documents)

        assert "processing_timestamp" in results[0]
        # Verify timestamp format
        timestamp = datetime.fromisoformat(
            results[0]["processing_timestamp"].replace("Z", "+00:00")
        )
        assert isinstance(timestamp, datetime)
