"""Integration tests for Knowledge Graph pipeline (NER + RE)."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import igraph as ig
import pytest

from src.components.data_ingestor import DataIngestor
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor


class TestKnowledgeGraphPipeline:
    """Integration tests for end-to-end Knowledge Graph pipeline."""

    @pytest.fixture
    def sample_corpus_data(self):
        """Create sample corpus data for testing."""
        return [
            {
                "idx": 0,
                "id": "600770",
                "title": "综艺股份",
                "text": """## 公司简称
综艺股份

## 公司代码
600770

## 主营业务
### 信息科技板块
* **子公司南京天悦**:
  * 核心业务: 超低功耗数模混合助听器芯片及高端数字语音处理技术的研发
  * 主要产品/服务: HA3950、HA330G、HA601SC、HA631SC芯片
  * 目标应用领域: 助听器市场""",
            },
            {
                "idx": 1,
                "id": "000001",
                "title": "平安银行",
                "text": """## 公司简称
平安银行

## 公司代码
000001

## 主营业务
### 零售业务
* 个人贷款
* 信用卡业务
* 财富管理""",
            },
        ]

    @pytest.fixture
    def corpus_file(self, sample_corpus_data):
        """Create a temporary corpus file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_corpus_data, f, ensure_ascii=False, indent=2)
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_full_knowledge_graph_pipeline(
        self, mock_openai_class, mock_settings_class, corpus_file
    ):
        """Test full pipeline from Data Ingestor to Knowledge Graph output."""
        # Mock settings for LLM adapter
        mock_settings = Mock()
        mock_settings.prompts_path = "config/prompts.yaml"
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create mock responses with proper structure
        def create_mock_response(content):
            response = Mock()
            response.choices = [Mock(message=Mock(content=content))]
            response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            return response

        # Mock API responses for NER and RE
        mock_client.chat.completions.create.side_effect = [
            # NER response for doc 1
            create_mock_response(
                '{"named_entities": ['
                '{"text": "综艺股份", "type": "COMPANY"}, '
                '{"text": "600770", "type": "COMPANY_CODE"}, '
                '{"text": "信息科技板块", "type": "BUSINESS_SEGMENT"}, '
                '{"text": "南京天悦", "type": "SUBSIDIARY"}, '
                '{"text": "助听器芯片", "type": "TECHNOLOGY"}, '
                '{"text": "HA3950", "type": "PRODUCT"}, '
                '{"text": "HA330G", "type": "PRODUCT"}, '
                '{"text": "HA601SC", "type": "PRODUCT"}, '
                '{"text": "HA631SC", "type": "PRODUCT"}'
                "]}"
            ),
            # RE response for doc 1
            create_mock_response(
                '{"triples": ['
                '["综艺股份", "公司代码是", "600770"], '
                '["综艺股份", "主营业务包括", "信息科技板块"], '
                '["南京天悦", "是子公司", "综艺股份"], '
                '["南京天悦", "核心业务是", "助听器芯片"], '
                '["南京天悦", "主要产品包括", "HA3950"], '
                '["南京天悦", "主要产品包括", "HA330G"], '
                '["南京天悦", "主要产品包括", "HA601SC"], '
                '["南京天悦", "主要产品包括", "HA631SC"]'
                "]}"
            ),
            # NER response for doc 2
            create_mock_response(
                '{"named_entities": ['
                '{"text": "平安银行", "type": "COMPANY"}, '
                '{"text": "000001", "type": "COMPANY_CODE"}, '
                '{"text": "零售业务", "type": "BUSINESS_SEGMENT"}, '
                '{"text": "个人贷款", "type": "PRODUCT"}, '
                '{"text": "信用卡业务", "type": "PRODUCT"}, '
                '{"text": "财富管理", "type": "CORE_BUSINESS"}'
                "]}"
            ),
            # RE response for doc 2
            create_mock_response(
                '{"triples": ['
                '["平安银行", "公司代码是", "000001"], '
                '["平安银行", "主营业务包括", "零售业务"], '
                '["零售业务", "包括", "个人贷款"], '
                '["零售业务", "包括", "信用卡业务"], '
                '["零售业务", "包括", "财富管理"]'
                "]}"
            ),
        ]

        # Step 1: Load corpus with Data Ingestor
        ingestor = DataIngestor()
        documents = ingestor.load_corpus(corpus_file)

        assert len(documents) == 2
        assert documents[0].idx == 0
        assert documents[0].title == "综艺股份"
        assert documents[1].idx == 1
        assert documents[1].title == "平安银行"

        # Convert to dict format for Knowledge Graph Constructor
        doc_dicts = [
            {"id": doc.idx, "text": doc.text, "title": doc.title} for doc in documents
        ]

        # Step 2: Process documents with Knowledge Graph Constructor
        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(doc_dicts)

        # Verify results structure
        assert len(results) == 2
        assert 0 in results
        assert 1 in results

        # Check results for first document
        assert "entities" in results[0]
        assert "triples" in results[0]
        assert len(results[0]["entities"]) == 9
        assert len(results[0]["triples"]) == 8

        # Verify graph structure
        assert isinstance(graph, ig.Graph)
        assert graph.is_directed()

        # Check vertices (should include all unique entities from both docs)
        vertex_names = {v["name"] for v in graph.vs}
        assert "综艺股份" in vertex_names
        assert "南京天悦" in vertex_names
        assert "平安银行" in vertex_names
        assert "600770" in vertex_names
        assert "000001" in vertex_names

        # Check entity types are preserved
        gqy_vertex = next(v for v in graph.vs if v["name"] == "综艺股份")
        assert gqy_vertex["entity_type"] == "COMPANY"

        # Check edges (relations)
        edge_count = graph.ecount()
        assert edge_count == 13  # 8 from doc1 + 5 from doc2

        # Verify specific relations exist
        edges = []
        for e in graph.es:
            source_name = graph.vs[e.source]["name"]
            target_name = graph.vs[e.target]["name"]
            relation = e["relation"]
            edges.append((source_name, relation, target_name))

        # Check some key relations
        assert ("综艺股份", "公司代码是", "600770") in edges
        assert ("南京天悦", "是子公司", "综艺股份") in edges
        assert ("平安银行", "公司代码是", "000001") in edges

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_graph_persistence_integration(
        self, mock_openai_class, mock_settings_class, corpus_file
    ):
        """Test graph saving and loading in integration scenario."""
        # Mock settings for LLM adapter
        mock_settings = Mock()
        mock_settings.prompts_path = "config/prompts.yaml"
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create mock responses with proper structure
        def create_mock_response(content):
            response = Mock()
            response.choices = [Mock(message=Mock(content=content))]
            response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            return response

        # Simple mock responses
        mock_client.chat.completions.create.side_effect = [
            create_mock_response(
                '{"named_entities": ['
                '{"text": "综艺股份", "type": "COMPANY"}, '
                '{"text": "600770", "type": "COMPANY_CODE"}'
                "]}"
            ),
            create_mock_response('{"triples": [["综艺股份", "公司代码是", "600770"]]}'),
            create_mock_response(
                '{"named_entities": ['
                '{"text": "平安银行", "type": "COMPANY"}, '
                '{"text": "000001", "type": "COMPANY_CODE"}'
                "]}"
            ),
            create_mock_response('{"triples": [["平安银行", "公司代码是", "000001"]]}'),
        ]

        # Load and process documents
        ingestor = DataIngestor()
        documents = ingestor.load_corpus(corpus_file)
        doc_dicts = [
            {"id": str(doc.idx), "text": doc.text, "title": doc.title}
            for doc in documents
        ]

        # Create first constructor and process
        constructor1 = KnowledgeGraphConstructor()
        results1, graph1 = constructor1.process_documents(doc_dicts)

        # Save graph to temporary location
        with tempfile.TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            graph_path = Path(tmp_dir) / "test_graph.graphml"

            # Save the graph
            assert constructor1.save_graph(graph_path) is True
            assert graph_path.exists()

            # Check metadata was saved
            metadata_path = Path(tmp_dir) / "test_graph_metadata.json"
            assert metadata_path.exists()

            # Load graph in new constructor
            constructor2 = KnowledgeGraphConstructor()
            loaded_graph = constructor2.load_graph(graph_path)

            assert loaded_graph is not None
            assert loaded_graph.vcount() == graph1.vcount()
            assert loaded_graph.ecount() == graph1.ecount()

            # Verify graph content preserved
            original_names = {v["name"] for v in graph1.vs}
            loaded_names = {v["name"] for v in loaded_graph.vs}
            assert original_names == loaded_names

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_batch_processing_integration(self, mock_openai_class, mock_settings_class):
        """Test batch processing with realistic data volume."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = "config/prompts.yaml"
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create 150 test documents
        test_docs = []
        for i in range(150):
            test_docs.append(
                {
                    "id": f"doc{i}",
                    "text": f"公司{i}是一家科技公司,代码是{i:06d}",
                    "title": f"公司{i}",
                }
            )

        # Create mock responses
        def create_mock_response(content):
            response = Mock()
            response.choices = [Mock(message=Mock(content=content))]
            response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            return response

        # Generate responses for all documents
        responses = []
        for i in range(150):
            # NER response
            responses.append(
                create_mock_response(
                    f'{{"named_entities": ['
                    f'{{"text": "公司{i}", "type": "COMPANY"}}, '
                    f'{{"text": "{i:06d}", "type": "COMPANY_CODE"}}'
                    f"]}}"
                )
            )
            # RE response
            responses.append(
                create_mock_response(
                    f'{{"triples": [["公司{i}", "公司代码是", "{i:06d}"]]}}'
                )
            )

        mock_client.chat.completions.create.side_effect = responses

        # Process with batch size of 50
        constructor = KnowledgeGraphConstructor()
        results, graph = constructor.process_documents(test_docs, batch_size=50)

        # Verify all documents processed
        assert len(results) == 150

        # Verify graph has correct structure
        assert graph.vcount() == 300  # 150 companies + 150 codes
        assert graph.ecount() == 150  # 150 relations

        # Check deduplication statistics
        stats = constructor.deduplication_stats
        assert stats["total_entities"] == 300
        assert stats["unique_entities"] == 300
        assert stats["merged_entities"] == 0  # No duplicates

        # Verify graph statistics
        graph_stats = constructor.calculate_graph_statistics()
        assert graph_stats["vertices_count"] == 300
        assert graph_stats["edges_count"] == 150
        assert (
            graph_stats["connected_components"] == 150
        )  # Each company-code pair is separate

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_concurrent_access_scenario(
        self, mock_openai_class, mock_settings_class, corpus_file
    ):
        """Test concurrent read/write scenarios."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = "config/prompts.yaml"
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Simple mock responses
        def create_mock_response(content):
            response = Mock()
            response.choices = [Mock(message=Mock(content=content))]
            response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            return response

        mock_client.chat.completions.create.side_effect = [
            create_mock_response(
                '{"named_entities": [{"text": "测试公司", "type": "COMPANY"}]}'
            ),
            create_mock_response('{"triples": []}'),
        ]

        # Process minimal data
        constructor = KnowledgeGraphConstructor()
        doc = [{"id": "test1", "text": "测试公司"}]
        results, graph = constructor.process_documents(doc)

        with tempfile.TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            graph_path = Path(tmp_dir) / "concurrent_test.graphml"

            # Save initial graph
            assert constructor.save_graph(graph_path) is True

            # Simulate concurrent operations
            # Save creates backup
            assert constructor.save_graph(graph_path) is True

            # Load while backup exists
            loaded_graph = constructor.load_graph(graph_path)
            assert loaded_graph is not None

            # Check backup was created
            backups = list(Path(tmp_dir).glob("graph_backup_*.graphml"))
            assert len(backups) > 0
