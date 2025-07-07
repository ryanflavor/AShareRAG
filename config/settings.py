import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys and Authentication
    deepseek_api_key: str | None = Field(default=None, description="DeepSeek API key")

    # DeepSeek Configuration
    deepseek_api_base: str = Field(
        default="https://api.deepseek.com/v1", description="Base URL for DeepSeek API"
    )
    deepseek_model: str = Field(
        default="deepseek-chat", description="DeepSeek model name"
    )

    # Model Configuration
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Name of the embedding model to use",
    )
    reranker_model_name: str = Field(
        default="Qwen/Qwen3-Reranker-4B",
        description="Name of the reranker model to use",
    )
    llm_model_name: str = Field(
        default="deepseek-chat", description="Name of the LLM model to use"
    )
    llm_adapter_type: str = Field(
        default="deepseek",
        description="Type of LLM adapter to use (deepseek, deepseek_reasoner, openai, etc.)",
    )

    # System Configuration
    batch_size: int = Field(default=32, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")
    log_level: str = Field(default="INFO", description="Logging level")

    # Storage Paths
    graph_storage_path: Path = Field(
        default=Path("output/graph"), description="Path to store graph data"
    )
    vector_storage_path: Path = Field(
        default=Path("output/vector_store"), description="Path to store vector data"
    )

    # Vector Storage Configuration
    vector_db_path: Path = Field(
        default=Path("./output/vector_store"), description="LanceDB storage location"
    )
    embedding_batch_size: int = Field(
        default=32, description="Batch size for embedding generation"
    )
    vector_table_name: str = Field(
        default="ashare_documents", description="LanceDB table name"
    )

    # Retrieval Configuration
    retriever_top_k: int = Field(
        default=10, description="Number of top documents to retrieve"
    )

    # Reranker Configuration
    reranker_relevance_threshold: float = Field(
        default=0.5, description="Minimum relevance score to keep documents"
    )
    reranker_batch_size: int = Field(default=8, description="Batch size for reranking")

    # Server Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    # Prompts Configuration
    prompts_path: Path = Field(
        default=Path("config/prompts.yaml"),
        description="Path to prompts configuration file",
    )

    # Knowledge Graph Configuration
    entity_type_priority: list[str] = Field(
        default=[
            "COMPANY",
            "PERSON",
            "LOCATION",
            "ORGANIZATION",
            "DATE",
            "MONEY",
            "PERCENT",
            "PRODUCT",
        ],
        description="Entity type priority for deduplication",
    )
    graph_pruning_threshold: int = Field(
        default=1_000_000,
        description="Maximum graph size before pruning",
    )
    processing_batch_size: int = Field(
        default=32,
        description="Batch size for document processing",
    )
    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for failed operations",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        description="Delay between retry attempts in seconds",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,  # Make settings immutable
        extra="ignore",
    )

    def __init__(self, **values):
        # Check if DOTENV_PATH is set for testing
        dotenv_path = os.environ.get("DOTENV_PATH")
        if dotenv_path:
            values["_env_file"] = dotenv_path
        super().__init__(**values)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v_upper

    @field_validator("api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator(
        "batch_size",
        "max_workers",
        "embedding_batch_size",
        "retriever_top_k",
        "reranker_batch_size",
    )
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate value is positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("reranker_relevance_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v


# Create global settings instance
settings = Settings()
