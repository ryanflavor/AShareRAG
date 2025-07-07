from unittest.mock import Mock

import pytest

from src.components.embedding_service import EmbeddingService
from src.components.vector_indexer import VectorIndexer
from src.components.vector_storage import VectorStorage


class TestVectorIndexer:
    """Unit tests for VectorIndexer component."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        mock = Mock(spec=EmbeddingService)
        return mock

    @pytest.fixture
    def mock_vector_storage(self):
        """Create a mock vector storage."""
        mock = Mock(spec=VectorStorage)
        return mock

    def test_vector_indexer_initialization(
        self, mock_embedding_service, mock_vector_storage
    ):
        """Test VectorIndexer can be initialized with required dependencies."""
        indexer = VectorIndexer(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        assert indexer.embedding_service == mock_embedding_service
        assert indexer.vector_storage == mock_vector_storage

    def test_index_documents_success(self, mock_embedding_service, mock_vector_storage):
        """Test successful document indexing."""
        # Arrange
        # Set up vector storage mock to have no table initially
        mock_vector_storage.table = None

        indexer = VectorIndexer(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        documents = [
            {
                "id": "doc1",
                "text": "Test content 1",
                "title": "Test Company 1",
                "source_file": "test1.json",
            },
            {
                "id": "doc2",
                "text": "Test content 2",
                "title": "Test Company 2",
                "source_file": "test2.json",
            },
        ]

        ner_re_results = {
            "doc1": {
                "entities": [
                    {"text": "Entity1", "type": "PERSON"},
                    {"text": "TestCo1", "type": "COMPANY"},
                ],
                "triples": [["Entity1", "works_for", "TestCo1"]],
            },
            "doc2": {
                "entities": [{"text": "Entity2", "type": "COMPANY"}],
                "triples": [["Entity2", "subsidiary_of", "TestCo1"]],
            },
        }

        # Mock the embedding service response
        mock_embedding_service.process_documents.return_value = [
            {
                "id": "doc1",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"source": "test1"},
            },
            {
                "id": "doc2",
                "embedding": [0.4, 0.5, 0.6],
                "metadata": {"source": "test2"},
            },
        ]

        # Act
        indexer.index_documents(documents, ner_re_results)

        # Assert
        mock_embedding_service.process_documents.assert_called_once()
        mock_vector_storage.add_documents.assert_called_once()
        mock_vector_storage.create_table.assert_called_once()

    def test_index_documents_with_empty_ner_results(
        self, mock_embedding_service, mock_vector_storage
    ):
        """Test handling of documents with no entities."""
        # Arrange
        mock_vector_storage.table = None
        indexer = VectorIndexer(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        documents = [{"id": "doc1", "text": "Test content 1"}]

        ner_re_results = {"doc1": {"entities": [], "triples": []}}

        mock_embedding_service.process_documents.return_value = [
            {"id": "doc1", "embedding": [0.1, 0.2, 0.3]}
        ]

        # Act
        indexer.index_documents(documents, ner_re_results)

        # Assert - should still process even with empty entities
        mock_embedding_service.process_documents.assert_called_once()

    def test_index_documents_storage_failure_retry(
        self, mock_embedding_service, mock_vector_storage
    ):
        """Test retry mechanism on storage failures."""
        # Arrange
        mock_vector_storage.table = None
        indexer = VectorIndexer(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        documents = [{"id": "doc1", "text": "Test content"}]
        ner_re_results = {"doc1": {"entities": [], "triples": []}}

        # Mock storage failure
        mock_embedding_service.process_documents.return_value = [
            {"id": "doc1", "embedding": [0.1, 0.2, 0.3]}
        ]
        mock_vector_storage.add_documents.side_effect = Exception("Storage error")

        # Act & Assert
        with pytest.raises(Exception, match="Storage error"):
            indexer.index_documents(documents, ner_re_results)

    def test_index_documents_invalid_document_format(
        self, mock_embedding_service, mock_vector_storage
    ):
        """Test handling of invalid document format."""
        # Arrange
        mock_vector_storage.table = None
        indexer = VectorIndexer(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        # Missing 'text' field
        documents = [{"id": "doc1", "content": "Wrong field name"}]

        ner_re_results = {"doc1": {"entities": [], "triples": []}}

        # Act
        indexer.index_documents(documents, ner_re_results)

        # Assert - should skip documents without 'text' field
        mock_embedding_service.process_documents.assert_not_called()

    def test_index_documents_missing_ner_results(
        self, mock_embedding_service, mock_vector_storage
    ):
        """Test handling of documents without corresponding NER results."""
        # Arrange
        mock_vector_storage.table = None
        indexer = VectorIndexer(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        documents = [
            {"id": "doc1", "text": "Test content"},
            {"id": "doc2", "text": "Test content 2"},
        ]

        # Only doc1 has NER results
        ner_re_results = {"doc1": {"entities": [], "triples": []}}

        mock_embedding_service.process_documents.return_value = [
            {"id": "doc1", "embedding": [0.1, 0.2, 0.3]}
        ]

        # Act
        indexer.index_documents(documents, ner_re_results)

        # Assert - should only process doc1
        mock_embedding_service.process_documents.assert_called_once()
        call_args = mock_embedding_service.process_documents.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["doc_id"] == "doc1"
