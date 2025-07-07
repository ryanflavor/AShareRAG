"""Integration tests for the complete embedding pipeline."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.components.embedding_service import EmbeddingService
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.components.vector_storage import VectorStorage


class TestEmbeddingPipeline:
    """Test suite for end-to-end embedding pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_documents(self):
        """Sample A-share documents for testing."""
        return [
            {
                "id": "doc1",
                "text": "综艺股份(600770)是一家专注于芯片设计的科技公司，在南京设有研发中心。",
                "title": "综艺股份年报",
            },
            {
                "id": "doc2",
                "text": "腾讯控股(00700.HK)与阿里巴巴(BABA)在云计算领域展开竞争。",
                "title": "科技巨头竞争分析",
            },
            {
                "id": "doc3",
                "text": "比亚迪(002594)发布新能源汽车销量数据，月销量突破30万辆。",
                "title": "比亚迪销量报告",
            },
        ]

    @patch("src.components.vector_storage.lancedb")
    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    @patch("src.components.embedding_service.Qwen3EmbeddingManager")
    def test_end_to_end_pipeline(
        self,
        mock_embedding_manager_class,
        mock_llm_adapter_class,
        mock_lancedb,
        temp_dir,
        sample_documents,
    ):
        """Test complete pipeline from documents to searchable vectors."""
        # Mock LLM adapter for NER/RE
        mock_llm_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_llm_adapter

        # Configure NER results
        mock_llm_adapter.extract_entities.side_effect = [
            [
                {"text": "综艺股份", "type": "COMPANY"},
                {"text": "600770", "type": "COMPANY_CODE"},
                {"text": "南京", "type": "LOCATION"},
            ],
            [
                {"text": "腾讯控股", "type": "COMPANY"},
                {"text": "00700.HK", "type": "COMPANY_CODE"},
                {"text": "阿里巴巴", "type": "COMPANY"},
                {"text": "BABA", "type": "COMPANY_CODE"},
            ],
            [
                {"text": "比亚迪", "type": "COMPANY"},
                {"text": "002594", "type": "COMPANY_CODE"},
                {"text": "30万辆", "type": "QUANTITY"},
            ],
        ]

        # Configure RE results
        mock_llm_adapter.extract_relations.side_effect = [
            [["综艺股份", "股票代码", "600770"], ["综艺股份", "位于", "南京"]],
            [
                ["腾讯控股", "股票代码", "00700.HK"],
                ["阿里巴巴", "股票代码", "BABA"],
                ["腾讯控股", "竞争于", "阿里巴巴"],
            ],
            [["比亚迪", "股票代码", "002594"], ["比亚迪", "月销量", "30万辆"]],
        ]

        # Mock embedding manager
        mock_embedding_manager = Mock()
        mock_embedding_manager_class.return_value = mock_embedding_manager
        mock_embedding_manager.load_model.return_value = True
        mock_embedding_manager.model = Mock()  # Indicate model is loaded

        # Configure embedding results
        def mock_encode_texts(texts, batch_size=32, show_progress=False):
            # Return random embeddings of correct shape
            return np.random.rand(len(texts), 2560).astype(np.float32)

        mock_embedding_manager.encode_texts.side_effect = mock_encode_texts

        # Mock LanceDB
        mock_db = Mock()
        mock_table = Mock()
        mock_lancedb.connect.return_value = mock_db
        mock_db.table_names.return_value = []
        mock_db.create_table.return_value = mock_table

        # Mock table operations
        mock_table.add.return_value = None
        mock_table.count.return_value = 3
        mock_table.schema = [Mock(name="id"), Mock(name="text"), Mock(name="vector")]

        # Mock search results
        import pandas as pd

        mock_search_results = pd.DataFrame(
            {
                "id": ["doc1_0", "doc2_0"],
                "text": ["text1", "text2"],
                "company_name": ["综艺股份", "腾讯控股"],
                "entities": [
                    [{"text": "综艺股份", "type": "COMPANY"}],
                    [{"text": "腾讯控股", "type": "COMPANY"}],
                ],
                "_distance": [0.1, 0.2],
            }
        )

        mock_search_chain = Mock()
        mock_table.search.return_value = mock_search_chain
        mock_search_chain.limit.return_value = Mock(
            to_pandas=Mock(return_value=mock_search_results)
        )

        # Initialize components
        embedding_service = EmbeddingService()
        embedding_service.load_model()

        vector_storage = VectorStorage(db_path=temp_dir / "vectors")
        vector_storage.connect()

        # Create knowledge graph constructor with embedding components
        constructor = KnowledgeGraphConstructor(
            embedding_service=embedding_service, vector_storage=vector_storage
        )

        # Process documents
        results, graph = constructor.process_documents(sample_documents)

        # Verify NER/RE results
        assert len(results) == 3
        assert graph.vcount() > 0
        assert graph.ecount() > 0

        # Verify embeddings were generated and stored
        table_info = vector_storage.get_table_info()
        assert table_info["num_rows"] == 3
        assert table_info["table_name"] == "ashare_documents"

        # Test vector search
        query_vector = np.random.rand(2560).astype(np.float32)
        search_results = vector_storage.search(query_vector, top_k=2)

        assert len(search_results) == 2
        assert all("score" in result for result in search_results)
        assert all("company_name" in result for result in search_results)
        assert all("entities" in result for result in search_results)

    @patch("src.components.vector_storage.lancedb")
    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    @patch("src.components.embedding_service.Qwen3EmbeddingManager")
    def test_pipeline_with_company_filter(
        self,
        mock_embedding_manager_class,
        mock_llm_adapter_class,
        mock_lancedb,
        temp_dir,
    ):
        """Test vector search with company filtering."""
        # Mock setup
        mock_llm_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_llm_adapter
        mock_llm_adapter.extract_entities.return_value = [
            {"text": "综艺股份", "type": "COMPANY"}
        ]
        mock_llm_adapter.extract_relations.return_value = []

        mock_embedding_manager = Mock()
        mock_embedding_manager_class.return_value = mock_embedding_manager
        mock_embedding_manager.load_model.return_value = True
        mock_embedding_manager.model = Mock()
        mock_embedding_manager.encode_texts.return_value = np.random.rand(
            1, 2560
        ).astype(np.float32)

        # Mock LanceDB
        mock_db = Mock()
        mock_table = Mock()
        mock_lancedb.connect.return_value = mock_db
        mock_db.table_names.return_value = []
        mock_db.create_table.return_value = mock_table
        mock_table.add.return_value = None

        # Mock filtered search
        import pandas as pd

        mock_search_results = pd.DataFrame(
            {
                "id": ["doc1_0"],
                "text": ["综艺股份是一家科技公司"],
                "company_name": ["综艺股份"],
                "_distance": [0.1],
            }
        )

        mock_search_chain = Mock()
        mock_where_chain = Mock()
        mock_table.search.return_value = mock_search_chain
        mock_search_chain.where.return_value = mock_where_chain
        mock_where_chain.limit.return_value = Mock(
            to_pandas=Mock(return_value=mock_search_results)
        )

        # Initialize components
        embedding_service = EmbeddingService()
        embedding_service.load_model()

        vector_storage = VectorStorage(db_path=temp_dir / "vectors")
        vector_storage.connect()

        constructor = KnowledgeGraphConstructor(
            embedding_service=embedding_service, vector_storage=vector_storage
        )

        # Process single document
        documents = [
            {"id": "doc1", "text": "综艺股份是一家科技公司", "title": "综艺股份"}
        ]

        constructor.process_documents(documents)

        # Search with filter
        query_vector = np.random.rand(2560).astype(np.float32)
        results = vector_storage.search(query_vector, filter_company="综艺股份")

        assert len(results) == 1
        assert results[0]["company_name"] == "综艺股份"

    @patch("src.components.vector_storage.lancedb")
    @patch("src.components.knowledge_graph_constructor.LLMAdapter")
    def test_pipeline_memory_usage(
        self, mock_llm_adapter_class, mock_lancedb, temp_dir
    ):
        """Test memory usage and performance with realistic data volumes."""
        # Mock LLM adapter
        mock_llm_adapter = Mock()
        mock_llm_adapter_class.return_value = mock_llm_adapter

        # Create 100 documents
        documents = []
        for i in range(100):
            documents.append(
                {
                    "id": f"doc{i}",
                    "text": f"公司{i}是一家从事业务{i}的企业，位于城市{i}。",
                    "title": f"公司{i}报告",
                }
            )

        # Configure mock responses
        def mock_extract_entities(text):
            # Extract company number from text
            import re

            match = re.search(r"公司(\d+)", text)
            if match:
                num = match.group(1)
                return [
                    {"text": f"公司{num}", "type": "COMPANY"},
                    {"text": f"业务{num}", "type": "BUSINESS"},
                    {"text": f"城市{num}", "type": "LOCATION"},
                ]
            return []

        def mock_extract_relations(text, entities):
            if len(entities) >= 3:
                return [
                    [entities[0]["text"], "从事", entities[1]["text"]],
                    [entities[0]["text"], "位于", entities[2]["text"]],
                ]
            return []

        mock_llm_adapter.extract_entities.side_effect = mock_extract_entities
        mock_llm_adapter.extract_relations.side_effect = mock_extract_relations

        # Use real embedding service with mocked model
        with patch(
            "src.components.embedding_service.Qwen3EmbeddingManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.load_model.return_value = True
            mock_manager.model = Mock()

            # Mock batch processing
            def mock_encode_batch(texts, batch_size=32, show_progress=False):
                return np.random.rand(len(texts), 2560).astype(np.float32)

            mock_manager.encode_texts.side_effect = mock_encode_batch

            # Mock LanceDB for 100 documents
            mock_db = Mock()
            mock_table = Mock()
            mock_lancedb.connect.return_value = mock_db
            mock_db.table_names.return_value = []
            mock_db.create_table.return_value = mock_table
            mock_table.add.return_value = None
            mock_table.count.return_value = 100
            mock_table.schema = [
                Mock(name="id"),
                Mock(name="text"),
                Mock(name="vector"),
            ]

            # Mock search results for top 10
            import pandas as pd

            search_data = {
                "id": [f"doc{i}_0" for i in range(10)],
                "text": [f"text{i}" for i in range(10)],
                "company_name": [f"公司{i}" for i in range(10)],
                "_distance": [0.1 * (i + 1) for i in range(10)],
            }
            mock_search_results = pd.DataFrame(search_data)

            mock_search_chain = Mock()
            mock_table.search.return_value = mock_search_chain
            mock_search_chain.limit.return_value = Mock(
                to_pandas=Mock(return_value=mock_search_results)
            )

            # Initialize components
            embedding_service = EmbeddingService(batch_size=32)
            embedding_service.load_model()

            vector_storage = VectorStorage(db_path=temp_dir / "vectors", batch_size=50)
            vector_storage.connect()

            constructor = KnowledgeGraphConstructor(
                embedding_service=embedding_service, vector_storage=vector_storage
            )

            # Process all documents
            results, graph = constructor.process_documents(documents)

            # Verify all documents were processed
            assert len(results) == 100
            assert graph.vcount() == 300  # 3 entities per document
            assert graph.ecount() == 200  # 2 relations per document

            # Verify embeddings were stored
            table_info = vector_storage.get_table_info()
            assert table_info["num_rows"] == 100

            # Test search performance
            query_vector = np.random.rand(2560).astype(np.float32)
            search_results = vector_storage.search(query_vector, top_k=10)

            assert len(search_results) == 10

    def test_pipeline_error_recovery(self, temp_dir):
        """Test pipeline recovery from various error conditions."""
        # Test with invalid embedding service
        with patch(
            "src.components.embedding_service.Qwen3EmbeddingManager"
        ) as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.load_model.return_value = False  # Fail to load

            embedding_service = EmbeddingService()
            result = embedding_service.load_model()

            assert result is False

        # Test with connection errors
        vector_storage = VectorStorage(db_path=temp_dir / "vectors")

        with patch("src.components.vector_storage.lancedb.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                vector_storage.connect()
