"""LLM adapter for Named Entity Recognition using DeepSeek V3."""

import json
import logging
import time
from pathlib import Path
from string import Template
from typing import TypedDict

import yaml
from openai import OpenAI

from config.settings import Settings

logger = logging.getLogger(__name__)

# Valid entity types from prompts.yaml
VALID_ENTITY_TYPES = {
    "COMPANY",
    "SUBSIDIARY",
    "AFFILIATE",
    "BUSINESS_SEGMENT",
    "CORE_BUSINESS",
    "PRODUCT",
    "TECHNOLOGY",
    "INDUSTRY_APPLICATION",
    "COMPANY_CODE",
}


class NamedEntity(TypedDict):
    """Type definition for named entity with text and type."""

    text: str
    type: str


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

    def extract_entities(
        self, text: str, max_retries: int = 3, include_types: bool = True
    ) -> list[str] | list[dict[str, str]]:
        """
        Extract named entities from text using LLM.

        Args:
            text: Input text to extract entities from
            max_retries: Maximum number of retry attempts for API failures
            include_types: If True, return entities with types; if False, return only text

        Returns:
            List of extracted entities. If include_types is True (default), returns
            list of dicts with 'text' and 'type' keys. If False, returns list of strings
            for backwards compatibility.

        Example:
            >>> adapter = LLMAdapter()
            >>> # Get typed entities (default)
            >>> entities = adapter.extract_entities("综艺股份是一家科技公司")
            >>> # Returns: [{"text": "综艺股份", "type": "COMPANY"}]
            >>>
            >>> # Get string entities (backwards compatible)
            >>> entities = adapter.extract_entities("综艺股份是一家科技公司", include_types=False)
            >>> # Returns: ["综艺股份"]
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
                entities = self._parse_response(response, include_types)
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

    def _parse_response(
        self, response: str, include_types: bool = True
    ) -> list[str] | list[dict[str, str]]:
        """
        Parse LLM response to extract entity list.

        Args:
            response: JSON string response from LLM
            include_types: If True, parse and validate typed entities; if False, return strings

        Returns:
            List of entities in requested format

        Raises:
            Logs errors for malformed JSON or invalid entity types
        """
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
                    if not entities:
                        return []

                    # Check if we have typed entities (dicts) or simple strings
                    first_entity = entities[0] if entities else None
                    if isinstance(first_entity, dict) and include_types:
                        # Process typed entities
                        valid_entities = []
                        for entity in entities:
                            if (
                                isinstance(entity, dict)
                                and "text" in entity
                                and "type" in entity
                                and entity["text"]
                                and str(entity["text"]).strip()
                                and entity["type"] in VALID_ENTITY_TYPES
                            ):
                                valid_entities.append(
                                    {
                                        "text": str(entity["text"]).strip(),
                                        "type": entity["type"],
                                    }
                                )
                            elif (
                                isinstance(entity, dict)
                                and "text" in entity
                                and "type" in entity
                            ):
                                # Log invalid entity type
                                logger.warning(
                                    f"Skipping entity with invalid type: {entity.get('type')}"
                                )
                        return valid_entities
                    else:
                        # Process as simple strings (backwards compatibility)
                        if include_types:
                            # Convert strings to typed format with default type
                            logger.warning(
                                "Received string entities when typed entities expected. "
                                "Consider updating the prompt to return typed entities."
                            )
                        return [
                            str(e).strip() for e in entities if e and str(e).strip()
                        ]

            logger.warning(f"Unexpected response format: {response}")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []
