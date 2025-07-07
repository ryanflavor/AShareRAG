"""LLM adapter for intent classification using DeepSeek."""

import json
import logging
from pathlib import Path
from string import Template
from typing import Any

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from src.adapters.deepseek_adapter import cache_response

logger = logging.getLogger(__name__)


class IntentClassificationAdapter:
    """Adapter for classifying query intent using DeepSeek LLM."""

    def __init__(self, enable_cache: bool = True):
        """Initialize the intent classification adapter."""
        self.settings = Settings()
        self.enable_cache = enable_cache

        # Set up cache
        if enable_cache:
            cache_dir = Path.cwd() / ".cache" / "intent"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file_name = str(cache_dir / "intent_cache.sqlite")

        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_api_base,
            max_retries=0,  # Use tenacity for retries
        )

        # Load prompts
        self._load_prompts()

        logger.info("IntentClassificationAdapter initialized")

    def _load_prompts(self):
        """Load prompts from configuration."""
        prompts_file = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
        with open(prompts_file, encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        self.intent_prompt = prompts.get("query_intent_classification", {})
        if not self.intent_prompt:
            raise ValueError(
                "query_intent_classification prompt not found in prompts.yaml"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _call_api(self, messages: list[dict]) -> str:
        """Call DeepSeek API with retry."""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

    @cache_response
    def classify_intent(self, query: str) -> dict[str, Any]:
        """Classify the intent of a query."""
        # Build messages
        messages = self._build_messages(query)

        # Call API
        response_text = self._call_api(messages)

        # Parse response
        try:
            result = json.loads(response_text)
            # Validate response format
            if not all(key in result for key in ["intent", "confidence", "reasoning"]):
                raise ValueError("Invalid response format")

            # Ensure intent is valid
            if result["intent"] not in ["fact_qa", "relationship_discovery"]:
                result["intent"] = "fact_qa"  # Default
                result["confidence"] = 0.5

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "intent": "fact_qa",
                "confidence": 0.0,
                "reasoning": f"Failed to parse response: {e!s}",
            }

    def _build_messages(self, query: str) -> list[dict]:
        """Build messages for the API call."""
        system_message = self.intent_prompt.get("system", "")
        user_template = Template(self.intent_prompt.get("template", ""))
        user_message = user_template.substitute(query=query)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
