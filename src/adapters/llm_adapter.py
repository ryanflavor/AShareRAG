"""Abstract base class for LLM adapters."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Import Settings for backward compatibility with tests

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM API call."""

    content: str
    model: str
    usage: dict[str, int]


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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text completion synchronously.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclass must implement generate method")

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 5000,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text completion asynchronously.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text
        """
        # Default implementation uses sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, max_tokens, temperature, top_p
        )

    def get_cache_stats(self) -> dict:
        """Get cache statistics (optional implementation)."""
        return {"cache_enabled": False}

    def clear_cache(self):
        """Clear cache (optional implementation)."""
        # Default implementation - subclasses can override
        return None
