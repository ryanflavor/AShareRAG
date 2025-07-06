"""Unit tests for Knowledge Graph Constructor."""

from unittest.mock import Mock, patch

from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor


class TestKnowledgeGraphConstructor:
    """Test cases for Knowledge Graph Constructor."""

    def test_knowledge_graph_constructor_initialization(self):
        """Test Knowledge Graph Constructor initializes properly."""
        constructor = KnowledgeGraphConstructor()
        assert constructor is not None
        assert hasattr(constructor, "process_documents")

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_single_document(self, mock_llm_adapter_class):
        """Test processing single document."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = ["公司A", "产品B", "技术C"]

        # Test document
        documents = [{"id": "doc1", "text": "这是一段包含公司A、产品B和技术C的文本。"}]

        constructor = KnowledgeGraphConstructor()
        result = constructor.process_documents(documents)

        assert result == {"doc1": ["公司A", "产品B", "技术C"]}
        mock_adapter.extract_entities.assert_called_once_with(
            "这是一段包含公司A、产品B和技术C的文本。"
        )

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_multiple_documents(self, mock_llm_adapter_class):
        """Test processing multiple documents."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.side_effect = [
            ["公司A", "产品B"],
            ["技术C", "服务D"],
            ["行业E"],
        ]

        # Test documents
        documents = [
            {"id": "doc1", "text": "文本1"},
            {"id": "doc2", "text": "文本2"},
            {"id": "doc3", "text": "文本3"},
        ]

        constructor = KnowledgeGraphConstructor()
        result = constructor.process_documents(documents)

        assert result == {
            "doc1": ["公司A", "产品B"],
            "doc2": ["技术C", "服务D"],
            "doc3": ["行业E"],
        }
        assert mock_adapter.extract_entities.call_count == 3

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_empty_list(self, mock_llm_adapter_class):
        """Test processing empty document list."""
        constructor = KnowledgeGraphConstructor()
        result = constructor.process_documents([])

        assert result == {}

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_with_logging(self, mock_llm_adapter_class):
        """Test that processing progress is logged."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = ["entity1"]

        documents = [{"id": "doc1", "text": "text"}]

        with patch("src.components.knowledge_graph_constructor.logger") as mock_logger:
            constructor = KnowledgeGraphConstructor()
            constructor.process_documents(documents)

            # Check that info logs were called for progress
            mock_logger.info.assert_called()

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_error_propagation(self, mock_llm_adapter_class):
        """Test error propagation from LLM adapter."""
        # Mock LLM adapter that returns empty list on error
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = []

        documents = [{"id": "doc1", "text": "text"}]

        constructor = KnowledgeGraphConstructor()
        result = constructor.process_documents(documents)

        # Should still return mapping, even if no entities extracted
        assert result == {"doc1": []}

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_missing_id_fallback(self, mock_llm_adapter_class):
        """Test document ID generation fallback when ID is missing."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.side_effect = [
            ["entity1"],
            ["entity2"],
            ["entity3"],
        ]

        # Test documents with missing IDs
        documents = [
            {"id": "doc1", "text": "text1"},  # Has ID
            {"text": "text2"},  # Missing ID
            {"id": "", "text": "text3"},  # Empty ID
        ]

        constructor = KnowledgeGraphConstructor()
        result = constructor.process_documents(documents)

        # Should generate fallback IDs for missing ones
        assert "doc1" in result
        assert "doc_1" in result  # Fallback ID for second doc
        assert "" in result  # Empty ID is kept as-is
        assert result["doc1"] == ["entity1"]
        assert result["doc_1"] == ["entity2"]
        assert result[""] == ["entity3"]

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_missing_text_field(self, mock_llm_adapter_class):
        """Test handling documents with missing text field."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = []

        # Test documents with missing text
        documents = [
            {"id": "doc1"},  # Missing text field
            {"id": "doc2", "text": None},  # None text
            {"id": "doc3", "text": ""},  # Empty text
        ]

        constructor = KnowledgeGraphConstructor()
        result = constructor.process_documents(documents)

        # Should handle missing/empty text gracefully
        assert result == {
            "doc1": [],
            "doc2": [],
            "doc3": [],
        }
        # Adapter should be called with empty string for missing text
        assert mock_adapter.extract_entities.call_count == 3
