"""Knowledge Graph Constructor component for NER orchestration."""

import logging

from src.adapters.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


class KnowledgeGraphConstructor:
    """Orchestrates Named Entity Recognition for building knowledge graphs."""

    def __init__(self):
        """Initialize Knowledge Graph Constructor with LLM adapter."""
        self.llm_adapter = LLMAdapter()

    def process_documents(self, documents: list[dict]) -> dict[str, list[str]]:
        """
        Process documents to extract named entities using LLM.

        Args:
            documents: List of documents with 'id' and 'text' fields

        Returns:
            Dictionary mapping document IDs to lists of extracted entities
        """
        if not documents:
            logger.info("No documents to process")
            return {}

        logger.info(f"Starting NER processing for {len(documents)} documents")

        results = {}

        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{idx}")
            text = doc.get("text", "")

            logger.info(f"Processing document {idx + 1}/{len(documents)}: {doc_id}")

            # Extract entities using LLM adapter
            entities = self.llm_adapter.extract_entities(text)

            # Store results
            results[doc_id] = entities

            logger.info(f"Extracted {len(entities)} entities from document {doc_id}")

        logger.info(
            f"Completed NER processing. Total documents processed: {len(documents)}"
        )

        return results
