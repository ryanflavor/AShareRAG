"""Unit tests for Knowledge Graph Constructor."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import igraph as ig
import numpy as np

from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor


class TestKnowledgeGraphConstructor:
    """Test cases for Knowledge Graph Constructor."""

    def test_knowledge_graph_constructor_initialization(self):
        """Test Knowledge Graph Constructor initializes properly."""
        constructor = KnowledgeGraphConstructor()
        assert constructor is not None
        assert hasattr(constructor, "process_documents")
        assert constructor.graph is None

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_single_document(self, mock_llm_adapter_class):
        """Test processing single document with NER and RE."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
            {"text": "技术C", "type": "TECHNOLOGY"},
        ]
        mock_adapter.extract_relations.return_value = [
            ["公司A", "生产", "产品B"],
            ["公司A", "拥有", "技术C"],
        ]

        # Test document
        documents = [{"id": "doc1", "text": "这是一段包含公司A、产品B和技术C的文本。"}]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Check results structure
        assert "doc1" in results
        assert "entities" in results["doc1"]
        assert "triples" in results["doc1"]
        assert len(results["doc1"]["entities"]) == 3
        assert len(results["doc1"]["triples"]) == 2

        # Check graph structure
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 3  # 3 unique entities
        assert graph.ecount() == 2  # 2 relations

        # Verify vertices
        vertex_names = [v["name"] for v in graph.vs]
        assert "公司A" in vertex_names
        assert "产品B" in vertex_names
        assert "技术C" in vertex_names

        mock_adapter.extract_entities.assert_called_once_with(
            "这是一段包含公司A、产品B和技术C的文本。"
        )

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_full_ner_re_flow(self, mock_llm_adapter_class):
        """Test full NER+RE processing flow with typed entities."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Define entities for doc1
        doc1_entities = [
            {"text": "综艺股份", "type": "COMPANY"},
            {"text": "600770", "type": "COMPANY_CODE"},
            {"text": "南京天悦", "type": "SUBSIDIARY"},
        ]

        # Define triples for doc1
        doc1_triples = [
            ["综艺股份", "公司代码是", "600770"],
            ["南京天悦", "是子公司", "综艺股份"],
        ]

        mock_adapter.extract_entities.return_value = doc1_entities
        mock_adapter.extract_relations.return_value = doc1_triples

        # Test document
        documents = [
            {"id": "doc1", "text": "综艺股份(600770)旗下的南京天悦是一家子公司。"}
        ]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Verify results structure
        assert results["doc1"]["entities"] == doc1_entities
        assert results["doc1"]["triples"] == doc1_triples

        # Verify extract_relations was called with typed entities
        mock_adapter.extract_relations.assert_called_once_with(
            "综艺股份(600770)旗下的南京天悦是一家子公司。", doc1_entities
        )

        # Verify graph structure
        assert graph.vcount() == 3
        assert graph.ecount() == 2

        # Check vertex attributes
        zongyi = graph.vs.find(name="综艺股份")
        assert zongyi["entity_type"] == "COMPANY"
        assert zongyi["first_seen"] == "doc1"

        nanjing = graph.vs.find(name="南京天悦")
        assert nanjing["entity_type"] == "SUBSIDIARY"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_empty_list(self, mock_llm_adapter_class):
        """Test processing empty document list."""
        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents([])

        assert results == {}
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 0
        assert graph.ecount() == 0

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_igraph_construction_from_triples(self, mock_llm_adapter_class):
        """Test igraph construction from triples."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Define test data
        entities = [
            {"text": "A", "type": "COMPANY"},
            {"text": "B", "type": "PRODUCT"},
            {"text": "C", "type": "TECHNOLOGY"},
        ]
        triples = [["A", "produces", "B"], ["A", "uses", "C"], ["B", "requires", "C"]]

        mock_adapter.extract_entities.return_value = entities
        mock_adapter.extract_relations.return_value = triples

        documents = [{"id": "doc1", "text": "test text"}]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Check graph has correct structure
        assert graph.vcount() == 3
        assert graph.ecount() == 3

        # Check edges
        edges = [
            (graph.vs[e.source]["name"], graph.vs[e.target]["name"], e["relation"])
            for e in graph.es
        ]
        assert ("A", "B", "produces") in edges
        assert ("A", "C", "uses") in edges
        assert ("B", "C", "requires") in edges

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_handling_of_duplicate_triples(self, mock_llm_adapter_class):
        """Test handling of duplicate triples."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        entities = [{"text": "A", "type": "COMPANY"}, {"text": "B", "type": "PRODUCT"}]

        # Return different triples for different documents
        mock_adapter.extract_entities.side_effect = [entities, entities]
        mock_adapter.extract_relations.side_effect = [
            [["A", "produces", "B"]],
            [["A", "produces", "B"]],  # Same triple from different doc
        ]

        documents = [{"id": "doc1", "text": "text1"}, {"id": "doc2", "text": "text2"}]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Should have only 2 vertices and 1 merged edge
        assert graph.vcount() == 2
        assert graph.ecount() == 1  # Edges are now merged with source tracking

        # Check that the edge has both source documents
        edge = graph.es[0]
        assert set(edge["source_docs"]) == {"doc1", "doc2"}

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_graph_vertex_deduplication(self, mock_llm_adapter_class):
        """Test graph vertex deduplication."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Return overlapping entities from different documents
        mock_adapter.extract_entities.side_effect = [
            [
                {"text": "公司A", "type": "COMPANY"},
                {"text": "产品B", "type": "PRODUCT"},
            ],
            [
                {"text": "公司A", "type": "COMPANY"},
                {"text": "技术C", "type": "TECHNOLOGY"},
            ],
        ]
        mock_adapter.extract_relations.side_effect = [
            [["公司A", "生产", "产品B"]],
            [["公司A", "研发", "技术C"]],
        ]

        documents = [{"id": "doc1", "text": "text1"}, {"id": "doc2", "text": "text2"}]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Should have only 3 unique vertices (公司A appears in both docs)
        assert graph.vcount() == 3
        vertex_names = [v["name"] for v in graph.vs]
        assert sorted(vertex_names) == ["产品B", "公司A", "技术C"]

        # Check that 公司A keeps its first seen doc
        company_a = graph.vs.find(name="公司A")
        assert company_a["first_seen"] == "doc1"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_self_referential_relations_handling(self, mock_llm_adapter_class):
        """Test handling of self-referential relations."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        entities = [{"text": "公司A", "type": "COMPANY"}]
        triples = [["公司A", "合并了", "公司A"]]  # Self-referential

        mock_adapter.extract_entities.return_value = entities
        mock_adapter.extract_relations.return_value = triples

        documents = [{"id": "doc1", "text": "公司A合并了自己的一个部门"}]

        with patch("src.components.knowledge_graph_constructor.logger") as mock_logger:
            constructor = KnowledgeGraphConstructor()
            results, graph = constructor.process_documents(documents)

            # Should create the self-loop edge
            assert graph.vcount() == 1
            assert graph.ecount() == 1

            # Check that it was logged
            mock_logger.debug.assert_called()

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_handling_of_empty_entity_list(self, mock_llm_adapter_class):
        """Test handling when no entities are extracted."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = []
        mock_adapter.extract_relations.return_value = []

        documents = [{"id": "doc1", "text": "这是一段没有实体的文本。"}]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        assert results["doc1"]["entities"] == []
        assert results["doc1"]["triples"] == []
        assert graph.vcount() == 0
        assert graph.ecount() == 0

        # extract_relations should not be called when no entities
        mock_adapter.extract_relations.assert_not_called()

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_entity_not_in_ner_results_handling(self, mock_llm_adapter_class):
        """Test handling when triple contains entity not in NER results."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # NER only extracts 公司A
        entities = [{"text": "公司A", "type": "COMPANY"}]
        # But RE returns triple with 产品B not in NER results
        triples = [["公司A", "生产", "产品B"]]

        mock_adapter.extract_entities.return_value = entities
        mock_adapter.extract_relations.return_value = triples

        documents = [{"id": "doc1", "text": "公司A生产产品B"}]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Should still create vertex for 产品B with UNKNOWN type
        assert graph.vcount() == 2

        product_b = graph.vs.find(name="产品B")
        assert product_b["entity_type"] == "UNKNOWN"
        assert product_b["first_seen"] == "doc1"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_with_embeddings(self, mock_llm_adapter_class):
        """Test processing documents with embedding generation."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
        ]
        mock_adapter.extract_relations.return_value = [["公司A", "生产", "产品B"]]

        # Mock embedding service
        mock_embedding_service = Mock()
        mock_processed_docs = [
            {
                "id": "doc1_0",
                "text": "test text",
                "vector": np.random.rand(2560).astype(np.float32),
                "company_name": "公司A",
                "doc_id": "doc1",
                "chunk_index": 0,
                "entities": [
                    {"text": "公司A", "type": "COMPANY"},
                    {"text": "产品B", "type": "PRODUCT"},
                ],
                "relations": [["公司A", "生产", "产品B"]],
                "relations_count": 1,
            }
        ]
        mock_embedding_service.process_documents.return_value = mock_processed_docs

        # Mock vector storage
        mock_vector_storage = Mock()
        mock_vector_storage.table = None

        # Test document
        documents = [{"id": "doc1", "text": "test text"}]

        # Create constructor with embedding components
        constructor = KnowledgeGraphConstructor(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )
        results, graph = constructor.process_documents(documents)

        # Verify NER/RE was performed
        assert "doc1" in results
        assert graph.vcount() == 2

        # Verify embedding service was called
        mock_embedding_service.process_documents.assert_called_once()
        call_args = mock_embedding_service.process_documents.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["doc_id"] == "doc1"
        assert call_args[0]["entities"] == [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
        ]

        # Verify vector storage was initialized and documents added
        mock_vector_storage.create_table.assert_called_once()
        mock_vector_storage.add_documents.assert_called_once_with(mock_processed_docs)

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_process_documents_without_embeddings(self, mock_llm_adapter_class):
        """Test processing documents without embedding components."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = [
            {"text": "公司A", "type": "COMPANY"}
        ]
        mock_adapter.extract_relations.return_value = []

        # Create constructor without embedding components
        constructor = KnowledgeGraphConstructor()

        documents = [{"id": "doc1", "text": "test text"}]
        results, graph = constructor.process_documents(documents)

        # Should still work without embeddings
        assert "doc1" in results
        assert graph.vcount() == 1

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_embedding_error_handling(self, mock_llm_adapter_class):
        """Test error handling when embedding fails."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter
        mock_adapter.extract_entities.return_value = [
            {"text": "公司A", "type": "COMPANY"}
        ]
        mock_adapter.extract_relations.return_value = []

        # Mock embedding service that fails
        mock_embedding_service = Mock()
        mock_embedding_service.process_documents.return_value = None

        # Mock vector storage
        mock_vector_storage = Mock()

        # Create constructor
        constructor = KnowledgeGraphConstructor(
            embedding_service=mock_embedding_service, vector_storage=mock_vector_storage
        )

        documents = [{"id": "doc1", "text": "test text"}]

        # Should not raise exception
        results, graph = constructor.process_documents(documents)

        # NER/RE should still succeed
        assert "doc1" in results
        assert graph.vcount() == 1

        # Vector storage should not be called
        mock_vector_storage.add_documents.assert_not_called()

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_prepare_documents_for_embedding(self, mock_llm_adapter_class):
        """Test document preparation for embedding."""
        # Mock LLM adapter
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()

        # Test documents with various metadata
        documents = [
            {"id": "doc1", "text": "text1", "title": "公司A年报"},
            {"id": "doc2", "text": "text2"},  # No title
            {"id": "doc3", "text": ""},  # Empty text
        ]

        ner_re_results = {
            "doc1": {
                "entities": [{"text": "公司A", "type": "COMPANY"}],
                "triples": [["公司A", "发布", "年报"]],
            },
            "doc2": {"entities": [{"text": "产品B", "type": "PRODUCT"}], "triples": []},
            "doc3": {"entities": [], "triples": []},
        }

        prepared_docs = constructor._prepare_documents_for_embedding(
            documents, ner_re_results
        )

        # Should only prepare docs with valid text
        assert len(prepared_docs) == 2

        # Check doc1 - has title
        assert prepared_docs[0]["doc_id"] == "doc1"
        assert prepared_docs[0]["company_name"] == "公司A年报"
        assert prepared_docs[0]["entities"] == [{"text": "公司A", "type": "COMPANY"}]
        assert prepared_docs[0]["relations"] == [["公司A", "发布", "年报"]]

        # Check doc2 - no title, no COMPANY entity
        assert prepared_docs[1]["doc_id"] == "doc2"
        assert prepared_docs[1]["company_name"] == "Unknown"
        assert prepared_docs[1]["entities"] == [{"text": "产品B", "type": "PRODUCT"}]

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_company_name_extraction_from_entities(self, mock_llm_adapter_class):
        """Test extracting company name from entities when no title."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()

        documents = [{"id": "doc1", "text": "text without title"}]

        ner_re_results = {
            "doc1": {
                "entities": [
                    {"text": "产品A", "type": "PRODUCT"},
                    {"text": "综艺股份", "type": "COMPANY"},  # Should use this
                    {"text": "技术B", "type": "TECHNOLOGY"},
                ],
                "triples": [],
            }
        }

        prepared_docs = constructor._prepare_documents_for_embedding(
            documents, ner_re_results
        )

        assert len(prepared_docs) == 1
        assert prepared_docs[0]["company_name"] == "综艺股份"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_save_graph_file_io_error(self, mock_llm_adapter_class):
        """Test error handling when saving graph fails due to file I/O error."""
        import os
        import tempfile
        from pathlib import Path

        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()
        constructor.graph = ig.Graph(directed=True)
        constructor.graph.add_vertex(name="test", entity_type="COMPANY")

        # Try to save to a read-only directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "readonly" / "graph.graphml"
            graph_path.parent.mkdir(exist_ok=True)
            # Make directory read-only
            os.chmod(graph_path.parent, 0o444)

            try:
                with patch(
                    "src.components.knowledge_graph_constructor.logger"
                ) as mock_logger:
                    result = constructor.save_graph(graph_path)
                    assert result is False
                    mock_logger.error.assert_called()
            finally:
                # Restore permissions for cleanup
                os.chmod(graph_path.parent, 0o755)

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_save_graph_disk_space_check(self, mock_llm_adapter_class):
        """Test disk space validation before saving graph."""

        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()
        constructor.graph = ig.Graph(directed=True)
        constructor.graph.add_vertex(name="test", entity_type="COMPANY")

        # Mock disk usage to simulate low disk space
        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value = MagicMock(free=1000)  # Only 1KB free

            with patch(
                "src.components.knowledge_graph_constructor.logger"
            ) as mock_logger:
                result = constructor.save_graph(Path("/tmp/test_graph.graphml"))
                assert result is False
                mock_logger.error.assert_called_with(
                    "Insufficient disk space. Required: ~10MB, Available: 0.00MB"
                )

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_load_graph_corrupted_file(self, mock_llm_adapter_class):
        """Test loading corrupted GraphML file."""
        import tempfile

        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()

        # Create a corrupted GraphML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".graphml", delete=False
        ) as f:
            f.write("<?xml version='1.0'?><corrupted>not valid graphml</corrupted>")
            temp_path = f.name

        try:
            with patch(
                "src.components.knowledge_graph_constructor.logger"
            ) as mock_logger:
                result = constructor.load_graph(Path(temp_path))
                assert result is None
                mock_logger.error.assert_called()
        finally:
            os.unlink(temp_path)

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_load_graph_permission_denied(self, mock_llm_adapter_class):
        """Test loading graph with permission denied."""
        import tempfile

        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()

        # Create a file with no read permissions
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".graphml", delete=False
        ) as f:
            f.write("<?xml version='1.0'?><graphml></graphml>")
            temp_path = f.name

        # Remove read permissions
        os.chmod(temp_path, 0o000)

        try:
            with patch(
                "src.components.knowledge_graph_constructor.logger"
            ) as mock_logger:
                result = constructor.load_graph(Path(temp_path))
                assert result is None
                mock_logger.error.assert_called()
        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_save_graph_with_retry_logic(self, mock_llm_adapter_class):
        """Test save with retry logic for transient failures."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()
        constructor.graph = ig.Graph(directed=True)
        constructor.graph.add_vertex(name="test", entity_type="COMPANY")

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "test_graph.graphml"

            # Mock write method to fail once then succeed
            with patch.object(ig.Graph, "write") as mock_write:
                # First call raises OSError, second succeeds
                def side_effect(path, format):
                    if mock_write.call_count == 1:
                        raise OSError("Temporary failure")
                    # Create a file with some content to simulate successful write
                    with open(path, "w") as f:
                        f.write('<?xml version="1.0"?><graphml></graphml>')

                mock_write.side_effect = side_effect

                with patch(
                    "src.components.knowledge_graph_constructor.logger"
                ) as mock_logger:
                    with patch("time.sleep"):  # Speed up test by not actually sleeping
                        result = constructor.save_graph(graph_path)
                        assert result is True
                        # Should have logged the retry
                        assert any(
                            "Retrying" in str(call)
                            for call in mock_logger.info.call_args_list
                        )

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_save_graph_xml_validation(self, mock_llm_adapter_class):
        """Test XML structure validation prevents injection."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()
        constructor.graph = ig.Graph(directed=True)

        # Try to add vertex with potentially malicious XML content
        malicious_name = '"><script>alert("xss")</script><x "'
        constructor.graph.add_vertex(name=malicious_name, entity_type="COMPANY")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "test_graph.graphml"

            # Save should succeed with proper escaping
            result = constructor.save_graph(graph_path)
            assert result is True

            # Verify the saved file has properly escaped content
            with open(graph_path) as f:
                content = f.read()
                assert "<script>" not in content
                assert "&lt;script&gt;" in content or "&quot;" in content

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_enhanced_entity_deduplication_with_metadata(self, mock_llm_adapter_class):
        """Test entity deduplication tracks occurrence count and preserves most specific type."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Doc1 has generic COMPANY type
        mock_adapter.extract_entities.side_effect = [
            [{"text": "综艺股份", "type": "COMPANY"}],
            [{"text": "综艺股份", "type": "LISTED_COMPANY"}],  # More specific type
            [{"text": "综艺股份", "type": "COMPANY"}],
        ]
        mock_adapter.extract_relations.side_effect = [[], [], []]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
        ]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Should have only one vertex
        assert graph.vcount() == 1

        # Check vertex attributes
        vertex = graph.vs.find(name="综艺股份")
        assert vertex["entity_type"] == "LISTED_COMPANY"  # Most specific type
        assert vertex["first_seen"] == "doc1"
        assert vertex["occurrence_count"] == 3
        assert set(vertex["source_docs"]) == {"doc1", "doc2", "doc3"}

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_enhanced_relation_merging_with_source_tracking(
        self, mock_llm_adapter_class
    ):
        """Test relation merging tracks all source documents."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        entities = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
        ]

        # Same relation from multiple documents
        mock_adapter.extract_entities.side_effect = [entities] * 3
        mock_adapter.extract_relations.side_effect = [
            [["公司A", "生产", "产品B"]],
            [["公司A", "生产", "产品B"]],
            [["公司A", "生产", "产品B"]],
        ]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
        ]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Should have 2 vertices
        assert graph.vcount() == 2

        # Check edges - should have one merged edge with all source docs
        edges = graph.es.select(relation="生产")
        assert len(edges) == 1

        edge = edges[0]
        assert edge["relation"] == "生产"
        assert set(edge["source_docs"]) == {"doc1", "doc2", "doc3"}
        assert edge["confidence"] == 1.0
        assert edge["first_seen"] == "doc1"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_entity_type_priority_resolution(self, mock_llm_adapter_class):
        """Test entity type resolution with priority rules."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Define type priority: LISTED_COMPANY > SUBSIDIARY > COMPANY > UNKNOWN
        mock_adapter.extract_entities.side_effect = [
            [{"text": "实体A", "type": "COMPANY"}],
            [{"text": "实体A", "type": "SUBSIDIARY"}],
            [{"text": "实体A", "type": "LISTED_COMPANY"}],
            [{"text": "实体A", "type": "UNKNOWN"}],
        ]
        mock_adapter.extract_relations.side_effect = [[], [], [], []]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
            {"id": "doc4", "text": "text4"},
        ]

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(documents)

        # Should keep LISTED_COMPANY as the most specific type
        vertex = graph.vs.find(name="实体A")
        assert vertex["entity_type"] == "LISTED_COMPANY"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_deduplication_statistics_tracking(self, mock_llm_adapter_class):
        """Test that deduplication statistics are tracked."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Multiple entities with duplicates
        mock_adapter.extract_entities.side_effect = [
            [{"text": "A", "type": "COMPANY"}, {"text": "B", "type": "PRODUCT"}],
            [{"text": "A", "type": "COMPANY"}, {"text": "C", "type": "TECHNOLOGY"}],
            [{"text": "B", "type": "PRODUCT"}, {"text": "C", "type": "TECHNOLOGY"}],
        ]

        mock_adapter.extract_relations.side_effect = [
            [["A", "produces", "B"]],
            [["A", "uses", "C"]],
            [["B", "requires", "C"]],
        ]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
        ]

        constructor = KnowledgeGraphConstructor()
        with patch("src.components.knowledge_graph_constructor.logger") as mock_logger:
            results, graph = constructor.process_documents(documents)

            # Should have deduplication statistics in logs
            assert graph.vcount() == 3  # A, B, C (deduplicated)

            # Check if statistics were logged
            log_messages = [str(call) for call in mock_logger.info.call_args_list]
            # Should log deduplication stats
            assert any("Deduplication statistics" in msg for msg in log_messages)

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_save_graph_with_default_path(self, mock_llm_adapter_class):
        """Test saving graph to default configured path."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create constructor and graph
        constructor = KnowledgeGraphConstructor()
        constructor.graph = ig.Graph(directed=True)
        constructor.graph.add_vertex(
            name="test", entity_type="COMPANY", occurrence_count=1, source_docs=["doc1"]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.components.knowledge_graph_constructor.Settings"
            ) as mock_settings:
                mock_settings.return_value.graph_storage_path = Path(tmp_dir) / "graph"

                # Save to default path
                result = constructor.save_graph()
                assert result is True

                # Check file was created
                expected_path = Path(tmp_dir) / "graph" / "knowledge_graph.graphml"
                assert expected_path.exists()

                # Check metadata was created
                metadata_path = (
                    Path(tmp_dir) / "graph" / "knowledge_graph_metadata.json"
                )
                assert metadata_path.exists()

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_load_graph_integration(self, mock_llm_adapter_class):
        """Test loading graph updates internal state correctly."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create and save a graph
        constructor1 = KnowledgeGraphConstructor()
        constructor1.graph = ig.Graph(directed=True)
        constructor1.graph.add_vertex(
            name="公司A",
            entity_type="COMPANY",
            occurrence_count=2,
            source_docs=["doc1", "doc2"],
        )
        constructor1.graph.add_vertex(
            name="产品B",
            entity_type="PRODUCT",
            occurrence_count=1,
            source_docs=["doc1"],
        )
        constructor1.graph.add_edge(
            0,
            1,
            relation="生产",
            source_docs=["doc1"],
            confidence=1.0,
            first_seen="doc1",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "test_graph.graphml"
            assert constructor1.save_graph(graph_path) is True

            # Load in new constructor
            constructor2 = KnowledgeGraphConstructor()
            loaded_graph = constructor2.load_graph(graph_path)

            assert loaded_graph is not None
            assert loaded_graph.vcount() == 2
            assert loaded_graph.ecount() == 1

            # Check vertex attributes preserved
            vertex_a = loaded_graph.vs.find(name="公司A")
            assert vertex_a["entity_type"] == "COMPANY"
            assert vertex_a["occurrence_count"] == 2
            # Note: GraphML may not preserve list types, so we just check the basics

            # Check edge attributes preserved
            edge = loaded_graph.es[0]
            assert edge["relation"] == "生产"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_backup_rotation(self, mock_llm_adapter_class):
        """Test backup rotation when saving graphs."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        constructor = KnowledgeGraphConstructor()
        constructor.graph = ig.Graph(directed=True)
        constructor.graph.add_vertex(
            name="test", entity_type="COMPANY", occurrence_count=1, source_docs=["doc1"]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            graph_path = Path(tmp_dir) / "knowledge_graph.graphml"

            # Save multiple times to test rotation
            for i in range(5):
                # Ensure file exists before creating backup
                if i > 0:
                    time.sleep(0.1)  # Ensure different timestamps
                assert constructor.save_graph(graph_path) is True

            # Check that only 3 backups exist (plus the main file = 4 total)
            backup_files = list(Path(tmp_dir).glob("graph_backup_*.graphml"))
            assert len(backup_files) <= 3, (
                f"Expected <= 3 backups, found {len(backup_files)}"
            )

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_graph_statistics_calculation(self, mock_llm_adapter_class):
        """Test graph statistics calculation."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create a complex graph
        entities = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "公司B", "type": "COMPANY"},
            {"text": "产品X", "type": "PRODUCT"},
            {"text": "产品Y", "type": "PRODUCT"},
            {"text": "技术Z", "type": "TECHNOLOGY"},
        ]

        triples = [
            ["公司A", "生产", "产品X"],
            ["公司A", "生产", "产品Y"],
            ["公司B", "生产", "产品Y"],
            ["产品X", "依赖", "技术Z"],
            ["产品Y", "依赖", "技术Z"],
        ]

        mock_adapter.extract_entities.return_value = entities
        mock_adapter.extract_relations.return_value = triples

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents([{"id": "doc1", "text": "test"}])

        # Get graph statistics
        stats = constructor.calculate_graph_statistics()

        assert stats["vertices_count"] == 5
        assert stats["edges_count"] == 5
        assert stats["density"] > 0
        assert stats["average_degree"] == 2.0
        assert stats["max_degree"] == 3
        assert "degree_distribution" in stats
        assert "entity_type_distribution" in stats
        assert stats["entity_type_distribution"]["COMPANY"] == 2
        assert stats["entity_type_distribution"]["PRODUCT"] == 2
        assert stats["entity_type_distribution"]["TECHNOLOGY"] == 1

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_top_entities_by_degree(self, mock_llm_adapter_class):
        """Test finding top entities by degree."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create a hub-and-spoke pattern
        entities = [{"text": f"Entity{i}", "type": "COMPANY"} for i in range(10)]

        # Entity0 is connected to all others
        triples = []
        for i in range(1, 10):
            triples.append(["Entity0", "关联", f"Entity{i}"])

        mock_adapter.extract_entities.return_value = entities
        mock_adapter.extract_relations.return_value = triples

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents([{"id": "doc1", "text": "test"}])

        # Get top entities
        top_entities = constructor.get_top_entities_by_degree(k=3)

        assert len(top_entities) == 3
        assert top_entities[0]["name"] == "Entity0"
        assert top_entities[0]["degree"] == 9
        assert top_entities[0]["entity_type"] == "COMPANY"

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_connected_components_analysis(self, mock_llm_adapter_class):
        """Test connected components analysis."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create two disconnected subgraphs
        entities = [
            {"text": "A1", "type": "COMPANY"},
            {"text": "A2", "type": "COMPANY"},
            {"text": "A3", "type": "COMPANY"},
            {"text": "B1", "type": "PRODUCT"},
            {"text": "B2", "type": "PRODUCT"},
        ]

        triples = [
            ["A1", "关联", "A2"],
            ["A2", "关联", "A3"],
            ["B1", "关联", "B2"],
        ]

        mock_adapter.extract_entities.return_value = entities
        mock_adapter.extract_relations.return_value = triples

        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents([{"id": "doc1", "text": "test"}])

        # Analyze components
        components_info = constructor.analyze_connected_components()

        assert components_info["total_components"] == 2
        assert components_info["largest_component_size"] == 3
        assert components_info["isolated_vertices"] == 0
        assert len(components_info["component_sizes"]) == 2

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_batch_processing(self, mock_llm_adapter_class):
        """Test batch processing for large document sets."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create 250 documents
        documents = [{"id": f"doc{i}", "text": f"text{i}"} for i in range(250)]

        # Mock different entities/relations for each doc
        def mock_extract_entities(text):
            doc_num = int(text.replace("text", ""))
            return [{"text": f"Entity{doc_num}", "type": "COMPANY"}]

        def mock_extract_relations(text, entities):
            doc_num = int(text.replace("text", ""))
            if doc_num > 0:
                return [[f"Entity{doc_num}", "关联", f"Entity{doc_num - 1}"]]
            return []

        mock_adapter.extract_entities.side_effect = mock_extract_entities
        mock_adapter.extract_relations.side_effect = mock_extract_relations

        constructor = KnowledgeGraphConstructor()

        with patch("src.components.knowledge_graph_constructor.gc") as mock_gc:
            results, graph = constructor.process_documents(documents, batch_size=100)

            # Check that gc.collect was called (for memory cleanup)
            assert mock_gc.collect.call_count >= 2  # At least after each batch

            # Check graph was built correctly
            assert graph.vcount() == 250
            assert graph.ecount() == 249  # Chain of connections

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_memory_monitoring(self, mock_llm_adapter_class):
        """Test memory monitoring during processing."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Simple entities/relations
        mock_adapter.extract_entities.return_value = [{"text": "A", "type": "COMPANY"}]
        mock_adapter.extract_relations.return_value = []

        constructor = KnowledgeGraphConstructor()

        # Mock psutil to simulate high memory usage
        with patch(
            "src.components.knowledge_graph_constructor.psutil.virtual_memory"
        ) as mock_memory:
            # Create a proper mock with all needed attributes
            mock_memory.return_value = MagicMock(
                percent=85.0,
                used=8 * 1024**3,  # 8GB
                total=10 * 1024**3,  # 10GB
            )

            with patch(
                "src.components.knowledge_graph_constructor.logger"
            ) as mock_logger:
                results, graph = constructor.process_documents(
                    [{"id": "doc1", "text": "test"}]
                )

                # Check that memory warning was logged
                warning_logged = any(
                    "High memory usage" in str(call)
                    for call in mock_logger.warning.call_args_list
                )
                if not warning_logged:
                    # Debug: print all warning calls
                    print("Warning calls:", mock_logger.warning.call_args_list)
                    print("All logger calls:")
                    for method in ["debug", "info", "warning", "error"]:
                        calls = getattr(mock_logger, method).call_args_list
                        if calls:
                            print(f"{method}: {calls}")
                assert warning_logged

    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_graph_pruning_for_large_graphs(self, mock_llm_adapter_class):
        """Test graph pruning when graph becomes too large."""
        mock_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_adapter

        # Create a graph that will trigger pruning
        all_entities = []
        all_triples = []

        # Create many isolated nodes and some connected ones
        for i in range(100):
            entities = [
                {"text": f"Isolated{i}", "type": "COMPANY"},
                {"text": f"Connected{i}", "type": "PRODUCT"},
            ]
            all_entities.extend(entities)

            # Only connect some entities
            if i < 50:
                all_triples.append([f"Connected{i}", "关联", f"Connected{i + 1}"])

        mock_adapter.extract_entities.return_value = all_entities
        mock_adapter.extract_relations.return_value = all_triples

        constructor = KnowledgeGraphConstructor()

        # Test with batch processing to trigger pruning
        with patch.object(constructor, "_should_prune_graph", return_value=True):
            # Process with batches to trigger pruning check
            results, graph = constructor.process_documents(
                [{"id": "doc1", "text": "test"}], batch_size=1
            )

            # The pruning implementation removed isolated vertices (degree < 2)
            # We had many isolated nodes, so count should be significantly reduced
            assert graph.vcount() < 200  # Should have pruned isolated vertices
            assert graph.vcount() > 0  # But should still have some vertices
