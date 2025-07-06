"""LLM adapter for Named Entity Recognition using DeepSeek V3."""

import json
import logging
import time
from pathlib import Path
from string import Template

import yaml
from openai import OpenAI

from config.settings import Settings

logger = logging.getLogger(__name__)


class LLMAdapter:
    """Adapter for LLM interactions, specifically for Named Entity Recognition."""

    def __init__(self):
        """Initialize LLM adapter with DeepSeek configuration."""
        self.settings = Settings()

        # Load prompts
        self._load_prompts()

        # Initialize OpenAI client with DeepSeek endpoint
        if self.settings.deepseek_api_key:
            self.client = OpenAI(
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_api_base,
            )
        else:
            logger.warning(
                "DeepSeek API key not found. LLM adapter will not be functional."
            )
            self.client = None

    def _load_prompts(self):
        """Load prompts from configuration file."""
        prompts_path = Path(self.settings.prompts_path)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found at {prompts_path}")

        with open(prompts_path, encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

        if "ner" not in self.prompts:
            raise ValueError("NER prompts not found in configuration")

        self.ner_config = self.prompts["ner"]

    def extract_entities(self, text: str, max_retries: int = 3) -> list[str]:
        """
        Extract named entities from text using LLM.

        Args:
            text: Input text to extract entities from
            max_retries: Maximum number of retry attempts for API failures

        Returns:
            List of extracted entity strings
        """
        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty entity list")
            return []

        if not self.client:
            logger.error("LLM client not initialized. Check API key configuration.")
            return []

        # Build messages for API call
        messages = self._build_messages(text)

        # Try to call API with retry logic
        for attempt in range(max_retries):
            try:
                response = self._call_llm(messages)
                entities = self._parse_response(response)
                logger.info(f"Successfully extracted {len(entities)} entities")
                return entities

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e!s}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"All {max_retries} attempts failed. Returning empty list."
                    )
                    return []

        return []

    def _build_messages(self, text: str) -> list[dict]:
        """Build message list for LLM API call."""
        messages = []

        # Add system prompt
        messages.append({"role": "system", "content": self.ner_config["system"]})

        # Add examples if provided
        if self.ner_config.get("examples"):
            for example in self.ner_config["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        # Add user query with text
        template = Template(self.ner_config["template"])
        user_content = template.substitute(passage=text)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM API and return response content."""
        response = self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=messages,
            temperature=0.1,  # Low temperature for consistent entity extraction
            max_tokens=2000,
        )

        return response.choices[0].message.content

    def _parse_response(self, response: str) -> list[str]:
        """Parse LLM response to extract entity list."""
        try:
            # Clean response and parse JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response)

            # Extract entities from expected format
            if isinstance(data, dict) and "named_entities" in data:
                entities = data["named_entities"]
                if isinstance(entities, list):
                    # Filter out empty strings and ensure all are strings
                    return [str(e).strip() for e in entities if e and str(e).strip()]

            logger.warning(f"Unexpected response format: {response}")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []
