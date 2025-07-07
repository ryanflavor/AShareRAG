"""Unit tests for Knowledge Graph Constructor."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import igraph as ig
import numpy as np
import pytest

from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.adapters.llm_adapter import LLMAdapter


class TestKnowledgeGraphConstructor:
    """Test cases for Knowledge Graph Constructor."""

    @pytest.fixture
    def mock_llm_adapter(self):
        """Create a mock LLM adapter for testing."""
        return Mock(spec=LLMAdapter)

    def test_dependency_injection_of_llm_adapter(self):
        """Test LLMAdapter is properly injected through constructor."""
        from src.adapters.llm_adapter import LLMAdapter

        # Create a mock LLM adapter
        mock_llm_adapter = Mock(spec=LLMAdapter)

        # Initialize with injected adapter
        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

        # Verify the adapter is properly set
        assert constructor.llm_adapter == mock_llm_adapter
        assert constructor.llm_adapter is not None

    def test_process_documents_without_embeddings(self):
        """Test that no embedding operations occur in process_documents."""
        from src.adapters.llm_adapter import LLMAdapter

        # Create mocks
        mock_llm_adapter = Mock(spec=LLMAdapter)
        mock_llm_adapter.extract_entities.return_value = [
            {"text": "Test Company", "type": "COMPANY"}
        ]
        mock_llm_adapter.extract_relations.return_value = [
            ["Test Company", "located_in", "Shanghai"]
        ]

        # Initialize without embedding services
        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

        # Process a document
        documents = [{"id": "doc1", "text": "Test Company is located in Shanghai."}]
        results, graph = constructor.process_documents(documents)

        # Verify that no embedding services are set
        assert (
            not hasattr(constructor, "embedding_service")
            or constructor.embedding_service is None
        )
        assert (
            not hasattr(constructor, "vector_storage")
            or constructor.vector_storage is None
        )

    def test_knowledge_graph_constructor_initialization(self):
        """Test Knowledge Graph Constructor initializes properly."""
        from src.adapters.llm_adapter import LLMAdapter

        mock_llm_adapter = Mock(spec=LLMAdapter)
        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        assert constructor is not None
        assert hasattr(constructor, "process_documents")
        assert constructor.graph is None

    def test_process_documents_single_document(self, mock_llm_adapter):
        """Test processing single document with NER and RE."""
        # Set up mock response
        mock_llm_adapter.extract_entities.return_value = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
            {"text": "技术C", "type": "TECHNOLOGY"},
        ]
        mock_llm_adapter.extract_relations.return_value = [
            ["公司A", "生产", "产品B"],
            ["公司A", "拥有", "技术C"],
        ]

        # Test document
        documents = [{"id": "doc1", "text": "这是一段包含公司A、产品B和技术C的文本。"}]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

        mock_llm_adapter.extract_entities.assert_called_once_with(
            "这是一段包含公司A、产品B和技术C的文本。"
        )

    def test_process_documents_full_ner_re_flow(self, mock_llm_adapter):
        """Test full NER+RE processing flow with typed entities."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter

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

        mock_llm_adapter.extract_entities.return_value = doc1_entities
        mock_llm_adapter.extract_relations.return_value = doc1_triples

        # Test document
        documents = [
            {"id": "doc1", "text": "综艺股份(600770)旗下的南京天悦是一家子公司。"}
        ]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        # Verify results structure
        assert results["doc1"]["entities"] == doc1_entities
        assert results["doc1"]["triples"] == doc1_triples

        # Verify extract_relations was called with typed entities
        mock_llm_adapter.extract_relations.assert_called_once_with(
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

    def test_process_documents_empty_list(self, mock_llm_adapter):
        """Test processing empty document list."""
        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents([])

        assert results == {}
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 0
        assert graph.ecount() == 0

    def test_igraph_construction_from_triples(self, mock_llm_adapter):
        """Test igraph construction from triples."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Define test data
        entities = [
            {"text": "A", "type": "COMPANY"},
            {"text": "B", "type": "PRODUCT"},
            {"text": "C", "type": "TECHNOLOGY"},
        ]
        triples = [["A", "produces", "B"], ["A", "uses", "C"], ["B", "requires", "C"]]

        mock_llm_adapter.extract_entities.return_value = entities
        mock_llm_adapter.extract_relations.return_value = triples

        documents = [{"id": "doc1", "text": "test text"}]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_handling_of_duplicate_triples(self, mock_llm_adapter):
        """Test handling of duplicate triples."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        entities = [{"text": "A", "type": "COMPANY"}, {"text": "B", "type": "PRODUCT"}]

        # Return different triples for different documents
        mock_llm_adapter.extract_entities.side_effect = [entities, entities]
        mock_llm_adapter.extract_relations.side_effect = [
            [["A", "produces", "B"]],
            [["A", "produces", "B"]],  # Same triple from different doc
        ]

        documents = [{"id": "doc1", "text": "text1"}, {"id": "doc2", "text": "text2"}]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        # Should have only 2 vertices and 1 merged edge
        assert graph.vcount() == 2
        assert graph.ecount() == 1  # Edges are now merged with source tracking

        # Check that the edge has both source documents
        edge = graph.es[0]
        assert set(edge["source_docs"]) == {"doc1", "doc2"}

    def test_graph_vertex_deduplication(self, mock_llm_adapter):
        """Test graph vertex deduplication."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Return overlapping entities from different documents
        mock_llm_adapter.extract_entities.side_effect = [
            [
                {"text": "公司A", "type": "COMPANY"},
                {"text": "产品B", "type": "PRODUCT"},
            ],
            [
                {"text": "公司A", "type": "COMPANY"},
                {"text": "技术C", "type": "TECHNOLOGY"},
            ],
        ]
        mock_llm_adapter.extract_relations.side_effect = [
            [["公司A", "生产", "产品B"]],
            [["公司A", "研发", "技术C"]],
        ]

        documents = [{"id": "doc1", "text": "text1"}, {"id": "doc2", "text": "text2"}]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        # Should have only 3 unique vertices (公司A appears in both docs)
        assert graph.vcount() == 3
        vertex_names = [v["name"] for v in graph.vs]
        assert sorted(vertex_names) == ["产品B", "公司A", "技术C"]

        # Check that 公司A keeps its first seen doc
        company_a = graph.vs.find(name="公司A")
        assert company_a["first_seen"] == "doc1"

    def test_self_referential_relations_handling(self, mock_llm_adapter):
        """Test handling of self-referential relations."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        entities = [{"text": "公司A", "type": "COMPANY"}]
        triples = [["公司A", "合并了", "公司A"]]  # Self-referential

        mock_llm_adapter.extract_entities.return_value = entities
        mock_llm_adapter.extract_relations.return_value = triples

        documents = [{"id": "doc1", "text": "公司A合并了自己的一个部门"}]

        with patch("src.components.knowledge_graph_constructor.logger") as mock_logger:
            constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
            results, graph = constructor.process_documents(documents)

            # Should create the self-loop edge
            assert graph.vcount() == 1
            assert graph.ecount() == 1

            # Check that it was logged
            mock_logger.debug.assert_called()

    def test_handling_of_empty_entity_list(self, mock_llm_adapter):
        """Test handling when no entities are extracted."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter
        mock_llm_adapter.extract_entities.return_value = []
        mock_llm_adapter.extract_relations.return_value = []

        documents = [{"id": "doc1", "text": "这是一段没有实体的文本。"}]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        assert results["doc1"]["entities"] == []
        assert results["doc1"]["triples"] == []
        assert graph.vcount() == 0
        assert graph.ecount() == 0

        # extract_relations should not be called when no entities
        mock_llm_adapter.extract_relations.assert_not_called()

    def test_entity_not_in_ner_results_handling(self, mock_llm_adapter):
        """Test handling when triple contains entity not in NER results."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # NER only extracts 公司A
        entities = [{"text": "公司A", "type": "COMPANY"}]
        # But RE returns triple with 产品B not in NER results
        triples = [["公司A", "生产", "产品B"]]

        mock_llm_adapter.extract_entities.return_value = entities
        mock_llm_adapter.extract_relations.return_value = triples

        documents = [{"id": "doc1", "text": "公司A生产产品B"}]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        # Should still create vertex for 产品B with UNKNOWN type
        assert graph.vcount() == 2

        product_b = graph.vs.find(name="产品B")
        assert product_b["entity_type"] == "UNKNOWN"
        assert product_b["first_seen"] == "doc1"

    def test_process_documents_without_embeddings(self, mock_llm_adapter):
        """Test processing documents without embedding components."""
        # Mock LLM adapter
        # Use injected mock
        # mock_adapter = mock_llm_adapter
        mock_llm_adapter.extract_entities.return_value = [
            {"text": "公司A", "type": "COMPANY"}
        ]
        mock_llm_adapter.extract_relations.return_value = []

        # Create constructor without embedding components
        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

        documents = [{"id": "doc1", "text": "test text"}]
        results, graph = constructor.process_documents(documents)

        # Should still work without embeddings
        assert "doc1" in results
        assert graph.vcount() == 1

    def test_save_graph_file_io_error(self, mock_llm_adapter):
        """Test error handling when saving graph fails due to file I/O error."""
        import os
        import tempfile
        from pathlib import Path

        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_save_graph_disk_space_check(self, mock_llm_adapter):
        """Test disk space validation before saving graph."""

        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_load_graph_corrupted_file(self, mock_llm_adapter):
        """Test loading corrupted GraphML file."""
        import tempfile

        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

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

    def test_load_graph_permission_denied(self, mock_llm_adapter):
        """Test loading graph with permission denied."""
        import tempfile

        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

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

    def test_save_graph_with_retry_logic(self, mock_llm_adapter):
        """Test save with retry logic for transient failures."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_save_graph_xml_validation(self, mock_llm_adapter):
        """Test XML structure validation prevents injection."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_enhanced_entity_deduplication_with_metadata(self, mock_llm_adapter):
        """Test entity deduplication tracks occurrence count and preserves most specific type."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Doc1 has generic COMPANY type
        mock_llm_adapter.extract_entities.side_effect = [
            [{"text": "综艺股份", "type": "COMPANY"}],
            [{"text": "综艺股份", "type": "LISTED_COMPANY"}],  # More specific type
            [{"text": "综艺股份", "type": "COMPANY"}],
        ]
        mock_llm_adapter.extract_relations.side_effect = [[], [], []]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
        ]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        # Should have only one vertex
        assert graph.vcount() == 1

        # Check vertex attributes
        vertex = graph.vs.find(name="综艺股份")
        assert vertex["entity_type"] == "LISTED_COMPANY"  # Most specific type
        assert vertex["first_seen"] == "doc1"
        assert vertex["occurrence_count"] == 3
        assert set(vertex["source_docs"]) == {"doc1", "doc2", "doc3"}

    def test_enhanced_relation_merging_with_source_tracking(self, mock_llm_adapter):
        """Test relation merging tracks all source documents."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        entities = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
        ]

        # Same relation from multiple documents
        mock_llm_adapter.extract_entities.side_effect = [entities] * 3
        mock_llm_adapter.extract_relations.side_effect = [
            [["公司A", "生产", "产品B"]],
            [["公司A", "生产", "产品B"]],
            [["公司A", "生产", "产品B"]],
        ]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
        ]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_entity_type_priority_resolution(self, mock_llm_adapter):
        """Test entity type resolution with priority rules."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Define type priority: LISTED_COMPANY > SUBSIDIARY > COMPANY > UNKNOWN
        mock_llm_adapter.extract_entities.side_effect = [
            [{"text": "实体A", "type": "COMPANY"}],
            [{"text": "实体A", "type": "SUBSIDIARY"}],
            [{"text": "实体A", "type": "LISTED_COMPANY"}],
            [{"text": "实体A", "type": "UNKNOWN"}],
        ]
        mock_llm_adapter.extract_relations.side_effect = [[], [], [], []]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
            {"id": "doc4", "text": "text4"},
        ]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents(documents)

        # Should keep LISTED_COMPANY as the most specific type
        vertex = graph.vs.find(name="实体A")
        assert vertex["entity_type"] == "LISTED_COMPANY"

    def test_deduplication_statistics_tracking(self, mock_llm_adapter):
        """Test that deduplication statistics are tracked."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Multiple entities with duplicates
        mock_llm_adapter.extract_entities.side_effect = [
            [{"text": "A", "type": "COMPANY"}, {"text": "B", "type": "PRODUCT"}],
            [{"text": "A", "type": "COMPANY"}, {"text": "C", "type": "TECHNOLOGY"}],
            [{"text": "B", "type": "PRODUCT"}, {"text": "C", "type": "TECHNOLOGY"}],
        ]

        mock_llm_adapter.extract_relations.side_effect = [
            [["A", "produces", "B"]],
            [["A", "uses", "C"]],
            [["B", "requires", "C"]],
        ]

        documents = [
            {"id": "doc1", "text": "text1"},
            {"id": "doc2", "text": "text2"},
            {"id": "doc3", "text": "text3"},
        ]

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        with patch("src.components.knowledge_graph_constructor.logger") as mock_logger:
            results, graph = constructor.process_documents(documents)

            # Should have deduplication statistics in logs
            assert graph.vcount() == 3  # A, B, C (deduplicated)

            # Check if statistics were logged
            log_messages = [str(call) for call in mock_logger.info.call_args_list]
            # Should log deduplication stats
            assert any("Deduplication statistics" in msg for msg in log_messages)

    def test_save_graph_with_default_path(self, mock_llm_adapter):
        """Test saving graph to default configured path."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Create constructor and graph
        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_load_graph_integration(self, mock_llm_adapter):
        """Test loading graph updates internal state correctly."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Create and save a graph
        constructor1 = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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
            constructor2 = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_backup_rotation(self, mock_llm_adapter):
        """Test backup rotation when saving graphs."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_graph_statistics_calculation(self, mock_llm_adapter):
        """Test graph statistics calculation."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

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

        mock_llm_adapter.extract_entities.return_value = entities
        mock_llm_adapter.extract_relations.return_value = triples

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
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

    def test_top_entities_by_degree(self, mock_llm_adapter):
        """Test finding top entities by degree."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Create a hub-and-spoke pattern
        entities = [{"text": f"Entity{i}", "type": "COMPANY"} for i in range(10)]

        # Entity0 is connected to all others
        triples = []
        for i in range(1, 10):
            triples.append(["Entity0", "关联", f"Entity{i}"])

        mock_llm_adapter.extract_entities.return_value = entities
        mock_llm_adapter.extract_relations.return_value = triples

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents([{"id": "doc1", "text": "test"}])

        # Get top entities
        top_entities = constructor.get_top_entities_by_degree(k=3)

        assert len(top_entities) == 3
        assert top_entities[0]["name"] == "Entity0"
        assert top_entities[0]["degree"] == 9
        assert top_entities[0]["entity_type"] == "COMPANY"

    def test_connected_components_analysis(self, mock_llm_adapter):
        """Test connected components analysis."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

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

        mock_llm_adapter.extract_entities.return_value = entities
        mock_llm_adapter.extract_relations.return_value = triples

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)
        results, graph = constructor.process_documents([{"id": "doc1", "text": "test"}])

        # Analyze components
        components_info = constructor.analyze_connected_components()

        assert components_info["total_components"] == 2
        assert components_info["largest_component_size"] == 3
        assert components_info["isolated_vertices"] == 0
        assert len(components_info["component_sizes"]) == 2

    def test_batch_processing(self, mock_llm_adapter):
        """Test batch processing for large document sets."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

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

        mock_llm_adapter.extract_entities.side_effect = mock_extract_entities
        mock_llm_adapter.extract_relations.side_effect = mock_extract_relations

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

        with patch("src.components.knowledge_graph_constructor.gc") as mock_gc:
            results, graph = constructor.process_documents(documents, batch_size=100)

            # Check that gc.collect was called (for memory cleanup)
            assert mock_gc.collect.call_count >= 2  # At least after each batch

            # Check graph was built correctly
            assert graph.vcount() == 250
            assert graph.ecount() == 249  # Chain of connections

    def test_memory_monitoring(self, mock_llm_adapter):
        """Test memory monitoring during processing."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

        # Simple entities/relations
        mock_llm_adapter.extract_entities.return_value = [
            {"text": "A", "type": "COMPANY"}
        ]
        mock_llm_adapter.extract_relations.return_value = []

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

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

    def test_graph_pruning_for_large_graphs(self, mock_llm_adapter):
        """Test graph pruning when graph becomes too large."""
        # Use injected mock
        # mock_adapter = mock_llm_adapter

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

        mock_llm_adapter.extract_entities.return_value = all_entities
        mock_llm_adapter.extract_relations.return_value = all_triples

        constructor = KnowledgeGraphConstructor(llm_adapter=mock_llm_adapter)

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
