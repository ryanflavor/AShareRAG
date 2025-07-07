"""Unit tests for DeepSeek adapter with performance optimizations."""

import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest
import yaml

from src.adapters.deepseek_adapter import DeepSeekAdapter


class TestDeepSeekAdapter:
    """Test cases for DeepSeek adapter with caching and HTTP optimizations."""

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
                    "relation_extraction": {
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

    @pytest.fixture
    def mock_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "llm")
            os.makedirs(cache_dir, exist_ok=True)
            yield tmpdir

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_adapter_initialization_with_cache(
        self, mock_openai_class, mock_settings_class, mock_prompts_file, mock_cache_dir
    ):
        """Test DeepSeek adapter initializes with cache enabled."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings_class.return_value = mock_settings

        # Mock os.getcwd to return temp directory
        with patch("os.getcwd", return_value=mock_cache_dir):
            adapter = DeepSeekAdapter(enable_cache=True, high_throughput=True)

            assert adapter is not None
            assert adapter.enable_cache is True
            assert hasattr(adapter, "cache_file_name")
            assert "llm_cache.sqlite" in adapter.cache_file_name

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    @patch("src.adapters.deepseek_adapter.httpx.Client")
    def test_high_throughput_http_client(
        self,
        mock_httpx_client,
        mock_openai_class,
        mock_settings_class,
        mock_prompts_file,
    ):
        """Test high-throughput HTTP client configuration."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings_class.return_value = mock_settings

        # Test with high_throughput=True
        adapter = DeepSeekAdapter(enable_cache=False, high_throughput=True)

        # Verify httpx.Client was called with correct parameters
        mock_httpx_client.assert_called_once()
        call_kwargs = mock_httpx_client.call_args[1]
        assert call_kwargs["limits"].max_connections == 500
        assert call_kwargs["limits"].max_keepalive_connections == 100
        assert call_kwargs["timeout"].read == 300.0

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_cache_hit_scenario(
        self, mock_openai_class, mock_settings_class, mock_prompts_file, mock_cache_dir
    ):
        """Test cache hit scenario for repeated API calls."""
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
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"named_entities": [{"text": "公司A", "type": "COMPANY"}]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("os.getcwd", return_value=mock_cache_dir):
            adapter = DeepSeekAdapter(enable_cache=True)

            # First call - should hit API
            result1 = adapter.extract_entities("测试文本")
            assert mock_client.chat.completions.create.call_count == 1

            # Second call with same text - should hit cache
            result2 = adapter.extract_entities("测试文本")
            assert (
                mock_client.chat.completions.create.call_count == 1
            )  # No additional API call

            # Results should be identical
            assert result1 == result2

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_cache_statistics(
        self, mock_openai_class, mock_settings_class, mock_prompts_file, mock_cache_dir
    ):
        """Test cache statistics functionality."""
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
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("os.getcwd", return_value=mock_cache_dir):
            adapter = DeepSeekAdapter(enable_cache=True)

            # Initial cache stats
            stats = adapter.get_cache_stats()
            assert stats["cache_enabled"] is True
            assert stats["entries"] == 0

            # Make some API calls
            adapter.extract_entities("文本1")
            adapter.extract_entities("文本2")

            # Check updated stats
            stats = adapter.get_cache_stats()
            assert stats["entries"] == 2

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_clear_cache(
        self, mock_openai_class, mock_settings_class, mock_prompts_file, mock_cache_dir
    ):
        """Test cache clearing functionality."""
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
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("os.getcwd", return_value=mock_cache_dir):
            adapter = DeepSeekAdapter(enable_cache=True)

            # Add entries to cache
            adapter.extract_entities("文本1")
            assert adapter.get_cache_stats()["entries"] > 0

            # Clear cache
            adapter.clear_cache()

            # Verify cache is cleared
            assert adapter.get_cache_stats()["entries"] == 0

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_retry_mechanism(
        self, mock_openai_class, mock_settings_class, mock_prompts_file
    ):
        """Test retry mechanism with exponential backoff."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with failures then success
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
                )
            )
        ]
        mock_client.chat.completions.create.side_effect = [
            Exception("API error 1"),
            Exception("API error 2"),
            mock_response,
        ]

        with patch("time.sleep"):  # Mock sleep to speed up test
            adapter = DeepSeekAdapter(enable_cache=False)
            result = adapter.extract_entities("测试文本")

            # Should succeed after retries
            assert result == [{"text": "entity1", "type": "COMPANY"}]
            assert mock_client.chat.completions.create.call_count == 3

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_performance_with_cache(
        self, mock_openai_class, mock_settings_class, mock_prompts_file, mock_cache_dir
    ):
        """Test performance improvement with caching."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with simulated delay
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        def slow_api_call(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms API latency
            mock_response = Mock()
            mock_response.choices = [
                Mock(
                    message=Mock(
                        content='{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
                    )
                )
            ]
            return mock_response

        mock_client.chat.completions.create = slow_api_call

        with patch("os.getcwd", return_value=mock_cache_dir):
            adapter = DeepSeekAdapter(enable_cache=True)

            # First call - slow (API)
            start = time.time()
            adapter.extract_entities("测试文本")
            first_call_time = time.time() - start

            # Second call - fast (cache)
            start = time.time()
            adapter.extract_entities("测试文本")
            second_call_time = time.time() - start

            # Cache should be at least 10x faster
            assert second_call_time < first_call_time / 10

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_concurrent_cache_access(
        self, mock_openai_class, mock_settings_class, mock_prompts_file, mock_cache_dir
    ):
        """Test thread-safe cache access with FileLock."""
        import queue
        import threading

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
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        with patch("os.getcwd", return_value=mock_cache_dir):
            adapter = DeepSeekAdapter(enable_cache=True)
            results = queue.Queue()

            def worker(text):
                try:
                    result = adapter.extract_entities(text)
                    results.put(("success", result))
                except Exception as e:
                    results.put(("error", str(e)))

            # Launch multiple threads
            threads = []
            for i in range(5):
                t = threading.Thread(target=worker, args=(f"文本{i}",))
                t.start()
                threads.append(t)

            # Wait for all threads
            for t in threads:
                t.join()

            # Check all operations succeeded
            success_count = 0
            while not results.empty():
                status, _ = results.get()
                if status == "success":
                    success_count += 1

            assert success_count == 5

    @patch("src.adapters.deepseek_adapter.Settings")
    def test_get_http_stats(self, mock_settings_class, mock_prompts_file):
        """Test HTTP client statistics retrieval."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.prompts_path = mock_prompts_file
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings_class.return_value = mock_settings

        adapter = DeepSeekAdapter(enable_cache=False, high_throughput=True)

        # Get HTTP stats
        stats = adapter.get_http_stats()
        assert isinstance(stats, dict)
        # Stats availability depends on httpx internal implementation
