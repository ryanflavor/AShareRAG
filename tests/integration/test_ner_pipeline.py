"""Integration tests for NER pipeline."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.components.data_ingestor import DataIngestor
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor


class TestNERPipeline:
    """Integration tests for end-to-end NER pipeline."""

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

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
    def test_full_ner_pipeline(
        self, mock_openai_class, mock_settings_class, corpus_file
    ):
        """Test full pipeline from Data Ingestor to NER output."""
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

        # Mock API responses for each document with typed entities
        mock_client.chat.completions.create.side_effect = [
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=""
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
                        )
                    )
                ]
            ),
            Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content=""
                            '{"named_entities": ['
                            '{"text": "平安银行", "type": "COMPANY"}, '
                            '{"text": "000001", "type": "COMPANY_CODE"}, '
                            '{"text": "零售业务", "type": "BUSINESS_SEGMENT"}, '
                            '{"text": "个人贷款", "type": "PRODUCT"}, '
                            '{"text": "信用卡业务", "type": "PRODUCT"}, '
                            '{"text": "财富管理", "type": "CORE_BUSINESS"}'
                            "]}"
                        )
                    )
                ]
            ),
        ]

        # Step 1: Load data using Data Ingestor
        ingestor = DataIngestor()
        documents = ingestor.load_corpus(corpus_file)

        assert len(documents) == 2
        assert documents[0].idx == 0
        assert documents[1].idx == 1

        # Convert Document objects to dict format expected by Knowledge Graph Constructor
        doc_dicts = [{"id": doc.idx, "text": doc.text} for doc in documents]

        # Step 2: Process documents with Knowledge Graph Constructor
        constructor = KnowledgeGraphConstructor()
        entity_mapping = constructor.process_documents(doc_dicts)

        # Verify results
        assert len(entity_mapping) == 2
        assert 0 in entity_mapping
        assert 1 in entity_mapping

        # Check entities for first document
        entities_0 = entity_mapping[0]
        assert isinstance(entities_0, list)
        assert len(entities_0) == 9

        # Check specific entities and their types
        entity_texts_0 = [e["text"] for e in entities_0]
        assert "综艺股份" in entity_texts_0
        assert "600770" in entity_texts_0
        assert "信息科技板块" in entity_texts_0
        assert "南京天悦" in entity_texts_0

        # Find specific entity and check its type
        company_entity = next(e for e in entities_0 if e["text"] == "综艺股份")
        assert company_entity["type"] == "COMPANY"

        code_entity = next(e for e in entities_0 if e["text"] == "600770")
        assert code_entity["type"] == "COMPANY_CODE"

        # Check entities for second document
        entities_1 = entity_mapping[1]
        assert isinstance(entities_1, list)
        assert len(entities_1) == 6

        entity_texts_1 = [e["text"] for e in entities_1]
        assert "平安银行" in entity_texts_1
        assert "000001" in entity_texts_1
        assert "零售业务" in entity_texts_1

        # Verify API was called twice
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
    def test_ner_pipeline_with_multiple_documents(
        self, mock_openai_class, mock_settings_class
    ):
        """Test NER pipeline with multiple documents."""
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

        # Create multiple test documents
        documents = [
            {"id": f"doc_{i}", "text": f"Test document {i} content"} for i in range(5)
        ]

        # Mock API responses with typed entities
        mock_client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=""
                        '{"named_entities": ['
                        '{"text": "entity1", "type": "COMPANY"}, '
                        '{"text": "entity2", "type": "PRODUCT"}'
                        "]}"
                    )
                )
            ]
        )

        # Process documents
        constructor = KnowledgeGraphConstructor()
        entity_mapping = constructor.process_documents(documents)

        # Verify all documents were processed
        assert len(entity_mapping) == 5
        for i in range(5):
            assert f"doc_{i}" in entity_mapping
            entities = entity_mapping[f"doc_{i}"]
            assert len(entities) == 2
            assert entities[0] == {"text": "entity1", "type": "COMPANY"}
            assert entities[1] == {"text": "entity2", "type": "PRODUCT"}

        # Verify API was called 5 times
        assert mock_client.chat.completions.create.call_count == 5

    @patch("src.adapters.llm_adapter.Settings")
    def test_ner_pipeline_without_api_key(self, mock_settings_class):
        """Test NER pipeline behavior when API key is not configured."""
        # Mock settings without API key
        mock_settings = Mock()
        mock_settings.prompts_path = "config/prompts.yaml"
        mock_settings.deepseek_api_key = None  # No API key
        mock_settings_class.return_value = mock_settings

        # Create test documents
        documents = [{"id": "doc1", "text": "Test content"}]

        # Process documents - should return empty entities
        constructor = KnowledgeGraphConstructor()
        entity_mapping = constructor.process_documents(documents)

        # Verify empty results due to missing API key
        assert entity_mapping == {"doc1": []}

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
    def test_ner_pipeline_with_typed_entities(
        self, mock_openai_class, mock_settings_class
    ):
        """Test NER pipeline specifically for typed entity extraction."""
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

        # Test document with various entity types
        documents = [
            {
                "id": "test_doc",
                "text": """综艺股份(600770)是一家信息科技公司。
                其子公司南京天悦主要研发助听器芯片,
                产品包括HA3950和HA330G等。
                另一家参股公司神州龙芯专注于集成电路设计。""",
            }
        ]

        # Mock API response with all entity types
        mock_client.chat.completions.create.return_value = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=""
                        '{"named_entities": ['
                        '{"text": "综艺股份", "type": "COMPANY"}, '
                        '{"text": "600770", "type": "COMPANY_CODE"}, '
                        '{"text": "南京天悦", "type": "SUBSIDIARY"}, '
                        '{"text": "神州龙芯", "type": "AFFILIATE"}, '
                        '{"text": "信息科技", "type": "BUSINESS_SEGMENT"}, '
                        '{"text": "助听器芯片", "type": "TECHNOLOGY"}, '
                        '{"text": "HA3950", "type": "PRODUCT"}, '
                        '{"text": "HA330G", "type": "PRODUCT"}, '
                        '{"text": "集成电路设计", "type": "CORE_BUSINESS"}, '
                        '{"text": "助听器市场", "type": "INDUSTRY_APPLICATION"}'
                        "]}"
                    )
                )
            ]
        )

        # Process documents
        constructor = KnowledgeGraphConstructor()
        entity_mapping = constructor.process_documents(documents)

        # Verify results
        assert "test_doc" in entity_mapping
        entities = entity_mapping["test_doc"]
        assert len(entities) == 10

        # Verify each entity type is present
        entity_types = {e["type"] for e in entities}
        expected_types = {
            "COMPANY",
            "COMPANY_CODE",
            "SUBSIDIARY",
            "AFFILIATE",
            "BUSINESS_SEGMENT",
            "TECHNOLOGY",
            "PRODUCT",
            "CORE_BUSINESS",
            "INDUSTRY_APPLICATION",
        }
        assert entity_types == expected_types

        # Verify specific entities
        company_entities = [e for e in entities if e["type"] == "COMPANY"]
        assert len(company_entities) == 1
        assert company_entities[0]["text"] == "综艺股份"

        product_entities = [e for e in entities if e["type"] == "PRODUCT"]
        assert len(product_entities) == 2
        product_texts = {e["text"] for e in product_entities}
        assert "HA3950" in product_texts
        assert "HA330G" in product_texts
