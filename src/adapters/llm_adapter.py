"""Abstract base class for LLM adapters."""

import logging
from abc import ABC, abstractmethod

# Import Settings for backward compatibility with tests
from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Abstract base class for all LLM adapters."""

    @abstractmethod
    def extract_entities(
        self, text: str, include_types: bool = True
    ) -> list[dict[str, str]]:
        """
        Extract named entities from text.

        Args:
            text: Input text
            include_types: Whether to include entity types

        Returns:
            List of entities with text and type
        """
        pass

    @abstractmethod
    def extract_relations(
        self, text: str, entities: list[dict[str, str]]
    ) -> list[list[str]]:
        """
        Extract relations between entities.

        Args:
            text: Input text
            entities: List of entities

        Returns:
            List of relation triples
        """
        pass

    def get_cache_stats(self) -> dict:
        """Get cache statistics (optional implementation)."""
        return {"cache_enabled": False}

    def clear_cache(self):
        """Clear cache (optional implementation)."""
        # Default implementation - subclasses can override
        return None
