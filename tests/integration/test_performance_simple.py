"""Simple performance tests that don't require API keys."""

import os
import time
import tempfile
from unittest.mock import patch, Mock

import pytest

from src.adapters.deepseek_adapter import DeepSeekAdapter


class TestSimplePerformance:
    """Simple performance tests with mocked API responses."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        mock_settings = Mock()
        mock_settings.deepseek_api_key = "test-key"
        mock_settings.deepseek_api_base = "https://api.deepseek.com/v1"
        mock_settings.deepseek_model = "deepseek-chat"
        mock_settings.prompts_path = "config/prompts.yaml"
        return mock_settings

    @pytest.fixture
    def mock_api_response(self):
        """Mock API response for testing."""

        def create_response(content):
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=content))]
            return mock_response

        return create_response

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_cache_performance(
        self, mock_openai_class, mock_settings_class, mock_settings, mock_api_response
    ):
        """Test that caching improves performance."""
        mock_settings_class.return_value = mock_settings

        # Mock OpenAI client with simulated delay
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        call_count = 0

        def slow_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate 100ms API latency
            return mock_api_response(
                '{"named_entities": [{"text": "测试公司", "type": "COMPANY"}]}'
            )

        mock_client.chat.completions.create = slow_api_call

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.getcwd", return_value=tmpdir):
                adapter = DeepSeekAdapter(enable_cache=True)

                # First call - slow
                start = time.time()
                result1 = adapter.extract_entities("测试文本")
                first_time = time.time() - start

                # Second call - should be cached
                start = time.time()
                result2 = adapter.extract_entities("测试文本")
                second_time = time.time() - start

                # Verify cache worked
                assert call_count == 1  # API called only once
                assert second_time < first_time / 10  # Cache is much faster
                assert result1 == result2  # Same results

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_http_connection_pooling(
        self, mock_openai_class, mock_settings_class, mock_settings
    ):
        """Test HTTP connection pooling configuration."""
        mock_settings_class.return_value = mock_settings

        # Track httpx.Client creation
        httpx_client_created = False
        original_httpx_client = None

        def mock_httpx_client(*args, **kwargs):
            nonlocal httpx_client_created, original_httpx_client
            httpx_client_created = True
            original_httpx_client = Mock()
            return original_httpx_client

        with patch(
            "src.adapters.deepseek_adapter.httpx.Client", side_effect=mock_httpx_client
        ):
            adapter = DeepSeekAdapter(enable_cache=False, high_throughput=True)

            # Verify high-throughput HTTP client was created
            assert httpx_client_created

            # Verify OpenAI client uses the custom HTTP client
            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args[1]
            assert call_kwargs["http_client"] == original_httpx_client

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_cache_statistics(
        self, mock_openai_class, mock_settings_class, mock_settings, mock_api_response
    ):
        """Test cache statistics tracking."""
        mock_settings_class.return_value = mock_settings

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_api_response(
            '{"named_entities": [{"text": "entity1", "type": "COMPANY"}]}'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.getcwd", return_value=tmpdir):
                adapter = DeepSeekAdapter(enable_cache=True)

                # Initial stats
                stats = adapter.get_cache_stats()
                assert stats["cache_enabled"] is True
                assert stats["entries"] == 0

                # Make some calls
                adapter.extract_entities("文本1")
                adapter.extract_entities("文本2")
                adapter.extract_entities("文本1")  # Duplicate

                # Check stats
                stats = adapter.get_cache_stats()
                assert stats["entries"] == 2  # Only unique entries

                # Clear cache
                adapter.clear_cache()
                stats = adapter.get_cache_stats()
                assert stats["entries"] == 0

    @patch("src.adapters.deepseek_adapter.Settings")
    @patch("src.adapters.deepseek_adapter.OpenAI")
    def test_concurrent_cache_safety(
        self, mock_openai_class, mock_settings_class, mock_settings, mock_api_response
    ):
        """Test thread-safe cache operations."""
        import threading
        import queue

        mock_settings_class.return_value = mock_settings

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Track calls per thread
        thread_calls = {}

        def thread_aware_api_call(*args, **kwargs):
            thread_id = threading.current_thread().ident
            thread_calls[thread_id] = thread_calls.get(thread_id, 0) + 1
            time.sleep(0.05)  # Small delay
            return mock_api_response(
                f'{{"named_entities": [{{"text": "entity{thread_id}", "type": "COMPANY"}}]}}'
            )

        mock_client.chat.completions.create = thread_aware_api_call

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.getcwd", return_value=tmpdir):
                adapter = DeepSeekAdapter(enable_cache=True)
                results = queue.Queue()

                def worker(text):
                    try:
                        result = adapter.extract_entities(text)
                        results.put(("success", len(result)))
                    except Exception as e:
                        results.put(("error", str(e)))

                # Launch multiple threads with same text
                threads = []
                for i in range(5):
                    t = threading.Thread(target=worker, args=("相同的文本",))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

                # Check results
                success_count = 0
                while not results.empty():
                    status, data = results.get()
                    if status == "success":
                        success_count += 1
                        assert data == 1  # Each result has 1 entity

                assert success_count == 5
                # Due to race conditions, multiple threads might make API calls
                # But it should be fewer than the total number of threads
                assert len(thread_calls) <= 3  # Allow some race conditions
