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
                    }
                },
                f,
            )
            temp_path = f.name

        yield temp_path

        # Cleanup
        os.unlink(temp_path)

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
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

        from src.adapters.llm_adapter import LLMAdapter

        adapter = LLMAdapter()
        assert adapter is not None
        assert hasattr(adapter, "extract_entities")

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
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

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(content='{"named_entities": ["公司A", "产品B", "技术C"]}')
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        from src.adapters.llm_adapter import LLMAdapter

        adapter = LLMAdapter()
        result = adapter.extract_entities("这是一段包含公司A、产品B和技术C的文本。")

        assert result == ["公司A", "产品B", "技术C"]
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
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

        from src.adapters.llm_adapter import LLMAdapter

        adapter = LLMAdapter()
        result = adapter.extract_entities("")
        assert result == []

    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
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

        from src.adapters.llm_adapter import LLMAdapter

        adapter = LLMAdapter()
        result = adapter.extract_entities("some text")
        assert result == []

    @patch("src.adapters.llm_adapter.time.sleep")
    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
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
            Mock(message=Mock(content='{"named_entities": ["entity1"]}'))
        ]
        mock_client.chat.completions.create.side_effect = [
            Exception("API error"),
            mock_response,
        ]

        from src.adapters.llm_adapter import LLMAdapter

        adapter = LLMAdapter()
        result = adapter.extract_entities("some text")

        assert result == ["entity1"]
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.adapters.llm_adapter.time.sleep")
    @patch("src.adapters.llm_adapter.Settings")
    @patch("src.adapters.llm_adapter.OpenAI")
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

        from src.adapters.llm_adapter import LLMAdapter

        adapter = LLMAdapter()
        result = adapter.extract_entities("some text")

        assert result == []
        assert (
            mock_client.chat.completions.create.call_count == 3
        )  # Default max retries

    @patch("src.adapters.llm_adapter.Settings")
    def test_missing_prompts_file(self, mock_settings_class):
        """Test adapter raises FileNotFoundError when prompts file is missing."""
        from src.adapters.llm_adapter import LLMAdapter

        # Mock settings to point to non-existent file
        mock_settings = Mock()
        mock_settings.prompts_path = "/non/existent/prompts.yaml"
        mock_settings.deepseek_api_key = "test-key"
        mock_settings_class.return_value = mock_settings

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Prompts file not found"):
            LLMAdapter()

    @patch("src.adapters.llm_adapter.Settings")
    def test_invalid_prompt_structure(self, mock_settings_class):
        """Test adapter raises ValueError when prompt structure is invalid."""
        from src.adapters.llm_adapter import LLMAdapter

        # Create prompts file without NER section
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"other_section": {"data": "value"}}, f)
            temp_path = f.name

        try:
            # Mock settings
            mock_settings = Mock()
            mock_settings.prompts_path = temp_path
            mock_settings.deepseek_api_key = "test-key"
            mock_settings_class.return_value = mock_settings

            # Should raise ValueError
            with pytest.raises(ValueError, match="NER prompts not found"):
                LLMAdapter()
        finally:
            os.unlink(temp_path)
