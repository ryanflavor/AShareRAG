"""Type definitions for intent detection module."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class IntentType(Enum):
    """Enumeration of query intent types."""

    FACT_QA = "fact_qa"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    UNKNOWN = "unknown"


@dataclass
class QueryIntent:
    """Represents the detected intent of a query."""

    query: str
    intent_type: IntentType
    confidence: float
    detection_method: str
    keywords_matched: list[str]
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )


@dataclass
class IntentDetectionConfig:
    """Configuration for intent detection."""

    keyword_threshold: float = 0.6
    llm_threshold: float = 0.7
    use_llm_fallback: bool = True
    llm_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout: float = 5.0
    cache_enabled: bool = True
    cache_ttl: int = 3600
