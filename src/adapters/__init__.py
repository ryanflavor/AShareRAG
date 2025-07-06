"""Adapters package for external service integrations."""

from config.settings import Settings

from .deepseek_adapter import DeepSeekAdapter
from .llm_adapter import LLMAdapter


def get_llm_adapter(**kwargs) -> LLMAdapter:
    """
    Factory function to get the appropriate LLM adapter based on configuration.

    Returns:
        LLMAdapter instance (currently DeepSeekAdapter by default)
    """
    settings = Settings()
    adapter_type = getattr(settings, "llm_adapter_type", "deepseek")

    if adapter_type == "deepseek":
        return DeepSeekAdapter(**kwargs)
    # Future adapters can be added here:
    # elif adapter_type == 'deepseek_reasoner':
    #     return DeepSeekReasonerAdapter(**kwargs)
    else:
        # Default to DeepSeek
        return DeepSeekAdapter(**kwargs)


# Export for backward compatibility
LLMAdapter = get_llm_adapter

__all__ = ["DeepSeekAdapter", "LLMAdapter", "get_llm_adapter"]
