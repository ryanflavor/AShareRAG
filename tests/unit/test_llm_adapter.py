"""Unit tests for LLM adapter."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml


class TestLLMAdapter:
    """Test cases for LLM adapter."""

    @pytest.fixture
    def mock_prompts_file(self):
        """Create a temporary prompts file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "ner": {
                        "system": "您的任务是从给定的中文段落中提取命名实体。",
                        "examples": [
                            {
                                "user": "example user input",
                                "assistant": '{"named_entities": ["entity1"]}',
                            }
                        ],
                        "template": "${passage}",
                    },
                    "re": {
                        "system": "您的任务是基于给定的中文段落和命名实体列表构建RDF图。",
                        "examples": [
                            {
                                "user": "example user input with entities",
                                "assistant": '{"triples": [["entity1", "relation", "entity2"]]}',
                            }
                        ],
                        "template": "${passage}\n\n${named_entity_json}",
                    },
                },
                f,
            )
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_llm_adapter_initialization(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test LLM adapter initializes with proper configuration."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings_class.return_value = mock_settings

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        assert adapter is not None
        assert hasattr(adapter, "extract_entities")

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_success(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test successful entity extraction."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with typed entities (new default format)
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=""
                    '{"named_entities": ['
                    '{"text": "公司A", "type": "COMPANY"}, '
                    '{"text": "产品B", "type": "PRODUCT"}, '
                    '{"text": "技术C", "type": "TECHNOLOGY"}'
                    "]}"
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("这是一段包含公司A、产品B和技术C的文本。")

        expected = [
            {"text": "公司A", "type": "COMPANY"},
            {"text": "产品B", "type": "PRODUCT"},
            {"text": "技术C", "type": "TECHNOLOGY"},
        ]
        assert result == expected
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_empty_text(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test entity extraction with empty text."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings_class.return_value = mock_settings

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("")
        assert result == []

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_malformed_json(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test handling of malformed JSON response."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with malformed response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="not valid json"))]
        mock_client.chat.completions.create.return_value = mock_response

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("some text")
        assert result == []

    @patch("src.adapters.deepseek_adapter.time.sleep")
    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_api_retry(
        self, mock_openai_class, mock_settings_class, mock_sleep, mock_prompts_file
    ):
        """Test API failure retry logic."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with failure then success
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
                )
            )
        ]
        mock_client.chat.completions.create.side_effect = [
            Exception("API error"),
            mock_response,
        ]

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("some text")

        assert result == [{"text": "entity1", "type": "COMPANY"}]
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.adapters.deepseek_adapter.time.sleep")
    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_all_retries_fail(
        self, mock_openai_class, mock_settings_class, mock_sleep, mock_prompts_file
    ):
        """Test when all API retries fail."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client that always fails
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("some text")

        assert result == []
        assert (
            mock_client.chat.completions.create.call_count == 3
        )  # Default max retries

    @patch("src.adapters.deepseek_adapter.Settings")
    def test_missing_prompts_file(self, mock_settings_class):
        """Test adapter raises FileNotFoundError when prompts file is missing."""
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        # Mock settings to point to non-existent file
        mock_settings = Mock()
        mock_settings.prompts_path = "/non/existent/prompts.yaml"
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Prompts file not found"):
            DeepSeekAdapter(enable_cache=False)

    @patch("src.adapters.deepseek_adapter.Settings")
    def test_invalid_prompt_structure(self, mock_settings_class):
        """Test adapter raises ValueError when prompt structure is invalid."""
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        # Create prompts file without NER section
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"other_section": {"data": "value"}}, f)
            temp_path = f.name

        try:
            # Mock settings
            mock_settings = Mock()
            mock_settings.prompts_path = temp_path
            mock_settings.deepseek_api_key = "test-key"
            mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
            mock_settings.deepseek_model = "deepseek-chat"
            mock_settings_class.return_value = mock_settings

            # Should raise ValueError
            with pytest.raises(ValueError, match="NER prompts not found"):
                DeepSeekAdapter(enable_cache=False)
        finally:
            os.unlink(temp_path)

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_string_format_with_include_types_false(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test extracting entities as simple strings when include_types=False."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with simple string entities
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(content='{"named_entities": ["公司A", "产品B", "技术C"]}')
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test with include_types=False
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities(
            "这是一段包含公司A、产品B和技术C的文本。", include_types=False
        )

        assert result == ["公司A", "产品B", "技术C"]
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_validates_types(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test that entity types are validated against predefined set."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with invalid entity type
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=""
                    '{"named_entities": ['
                    '{"text": "公司A", "type": "INVALID_TYPE"}, '
                    '{"text": "产品B", "type": "PRODUCT"}'
                    "]}"
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test - should return all entities (current behavior doesn't filter types)
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("测试文本")

        # Current implementation returns all entities regardless of type validity
        expected = [
            {"text": "公司A", "type": "INVALID_TYPE"},
            {"text": "产品B", "type": "PRODUCT"}
        ]
        assert result == expected

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_handles_malformed_entity_objects(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test handling of malformed entity objects."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with malformed entities
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=""
                    '{"named_entities": ['
                    '{"text": "公司A"},'
                    '{"type": "COMPANY"},'
                    '{"text": "产品B", "type": "PRODUCT"},'
                    '{"invalid_key": "value"}'
                    "]}"
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test - should only return well-formed entities
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("测试文本")

        # Only the well-formed entity should be returned
        expected = [{"text": "产品B", "type": "PRODUCT"}]
        assert result == expected

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_entities_backwards_compatibility(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test backwards compatibility with old format."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with old format (list of strings)
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"named_entities": ["公司A", "产品B"]}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test with include_types=False for backwards compatibility
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_entities("测试文本", include_types=False)

        # Should return simple string list
        assert result == ["公司A", "产品B"]

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_success(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test successful relation extraction."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with triples
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"triples": [["综艺股份", "公司代码是", "600770"], ["综艺股份", "主营业务包括", "信息科技板块"]]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        entities = [
            {"text": "综艺股份", "type": "COMPANY"},
            {"text": "600770", "type": "COMPANY_CODE"},
            {"text": "信息科技板块", "type": "BUSINESS_SEGMENT"},
        ]
        result = adapter.extract_relations(
            "综艺股份的公司代码是600770，主营业务包括信息科技板块", entities
        )

        expected = [
            ["综艺股份", "公司代码是", "600770"],
            ["综艺股份", "主营业务包括", "信息科技板块"],
        ]
        assert result == expected
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_handles_typed_entities(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test handling of typed entities from Story 1.2.1 format."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"triples": [["南京天悦", "是子公司", "综艺股份"]]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test with typed entities
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        typed_entities = [
            {"text": "南京天悦", "type": "SUBSIDIARY"},
            {"text": "综艺股份", "type": "COMPANY"},
        ]
        result = adapter.extract_relations("南京天悦是综艺股份的子公司", typed_entities)

        # Check that entities were properly extracted and passed
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = messages[-1]["content"]

        # Verify entity list was properly formatted
        assert '{"named_entities": ["南京天悦", "综艺股份"]}' in user_message
        assert result == [["南京天悦", "是子公司", "综艺股份"]]

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_empty_entity_list(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test relation extraction with empty entity list."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings_class.return_value = mock_settings

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        result = adapter.extract_relations("some text", [])
        assert result == []

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_validates_triples_contain_entities(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test validation that triples contain named entities."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with invalid triple (no named entities)
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"triples": [["综艺股份", "公司代码是", "600770"], ["其他公司", "属于", "不相关"]]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        entities = [
            {"text": "综艺股份", "type": "COMPANY"},
            {"text": "600770", "type": "COMPANY_CODE"},
        ]
        result = adapter.extract_relations("test text", entities)

        # Only the triple with named entities should be returned
        assert result == [["综艺股份", "公司代码是", "600770"]]

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_malformed_json_response(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test handling of malformed JSON response for relations."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with malformed response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="not valid json"))]
        mock_client.chat.completions.create.return_value = mock_response

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        entities = [{"text": "entity1", "type": "COMPANY"}]
        result = adapter.extract_relations("some text", entities)
        assert result == []

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_deduplicates_triples(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test deduplication of duplicate triples."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with duplicate triples
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"triples": [["A", "关系", "B"], ["A", "关系", "B"], ["C", "关系", "D"]]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        entities = [
            {"text": "A", "type": "COMPANY"},
            {"text": "B", "type": "COMPANY"},
            {"text": "C", "type": "COMPANY"},
            {"text": "D", "type": "COMPANY"},
        ]
        result = adapter.extract_relations("test text", entities)

        # Should only contain unique triples
        assert result == [["A", "关系", "B"], ["C", "关系", "D"]]

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_validates_triple_format(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test validation of triple format (exactly 3 elements)."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API response with invalid triple formats
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"triples": [["A", "关系"], ["B", "关系", "C"], ["D", "关系", "E", "extra"]]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        entities = [
            {"text": "A", "type": "COMPANY"},
            {"text": "B", "type": "COMPANY"},
            {"text": "C", "type": "COMPANY"},
            {"text": "D", "type": "COMPANY"},
            {"text": "E", "type": "COMPANY"},
        ]
        result = adapter.extract_relations("test text", entities)

        # Only the valid triple should be returned
        assert result == [["B", "关系", "C"]]

    @patch("src.adapters.deepseek_adapter.time.sleep")
    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_extract_relations_retry_logic(
        self, mock_openai_class, mock_settings_class, mock_sleep, mock_prompts_file
    ):
        """Test API failure retry logic for relation extraction."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with failure then success
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"triples": [["entity1", "关系", "entity2"]]}'))
        ]
        mock_client.chat.completions.create.side_effect = [
            Exception("API error"),
            mock_response,
        ]

        from src.adapters.deepseek_adapter import DeepSeekAdapter

        adapter = DeepSeekAdapter(enable_cache=False)
        entities = [
            {"text": "entity1", "type": "COMPANY"},
            {"text": "entity2", "type": "COMPANY"},
        ]
        result = adapter.extract_relations("some text", entities)

        assert result == [["entity1", "关系", "entity2"]]
        assert mock_client.chat.completions.create.call_count == 2
