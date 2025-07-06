import os
from pathlib import Path

import pytest

from config.settings import Settings


class TestSettings:
    """Test the configuration settings system."""

    def test_load_settings_from_env(self, test_env_file):
        """Test loading settings from environment file."""
        # Set the environment variable to point to our test .env file
        os.environ["DOTENV_PATH"] = test_env_file

        settings = Settings()

        # Test API keys
        assert settings.deepseek_api_key == "test_deepseek_key"

        # Test model configuration
        assert settings.embedding_model_name == "test/embedding-model"
        assert settings.reranker_model_name == "test/reranker-model"
        assert settings.llm_model_name == "test-llm"

        # Test system configuration
        assert settings.batch_size == 16
        assert settings.max_workers == 2
        assert settings.log_level == "DEBUG"

        # Test storage paths
        assert settings.graph_storage_path == Path("test/graph")
        assert settings.vector_storage_path == Path("test/vector")

        # Test server configuration
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8001

    def test_default_settings(self):
        """Test default settings when no env vars are set."""
        # Clear relevant environment variables
        env_vars_to_clear = [
            "DEEPSEEK_API_KEY",
            "EMBEDDING_MODEL_NAME",
            "RERANKER_MODEL_NAME",
            "LLM_MODEL_NAME",
            "BATCH_SIZE",
            "MAX_WORKERS",
            "LOG_LEVEL",
            "GRAPH_STORAGE_PATH",
            "VECTOR_STORAGE_PATH",
            "API_HOST",
            "API_PORT",
            "DOTENV_PATH",
        ]
        for var in env_vars_to_clear:
            os.environ.pop(var, None)

        settings = Settings()

        # Test default values
        assert settings.embedding_model_name == "Qwen/Qwen3-Embedding-4B"
        assert settings.reranker_model_name == "Qwen/Qwen3-Reranker-4B"
        assert settings.llm_model_name == "deepseek-chat"
        assert settings.batch_size == 32
        assert settings.max_workers == 4
        assert settings.log_level == "INFO"
        assert settings.graph_storage_path == Path("output/graph")
        assert settings.vector_storage_path == Path("output/vector_store")
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_settings_immutable(self, test_env_file):
        """Test that settings are immutable after creation."""
        os.environ["DOTENV_PATH"] = test_env_file
        settings = Settings()

        # Attempt to modify settings should raise an error
        with pytest.raises(ValueError):  # Settings are frozen
            settings.batch_size = 64

    def test_invalid_data_types(self):
        """Test that invalid data types raise appropriate errors."""
        os.environ["BATCH_SIZE"] = "not_a_number"
        os.environ["API_PORT"] = "invalid_port"

        with pytest.raises(ValueError):
            Settings()

    def test_prompts_yaml_path_exists(self):
        """Test that prompts.yaml path is correctly configured."""
        # Clean up any leftover env vars
        os.environ.pop("BATCH_SIZE", None)
        os.environ.pop("API_PORT", None)

        settings = Settings()
        assert settings.prompts_path == Path("config/prompts.yaml")
