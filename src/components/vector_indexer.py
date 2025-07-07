"""Vector indexer component for generating embeddings and storing in vector database."""

import logging
from typing import Any, Dict, List

from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Handles embedding generation and vector storage for documents."""

    def __init__(
        self, embedding_service: EmbeddingService, vector_storage: VectorStorage
    ):
        """Initialize VectorIndexer with injected dependencies.

        Args:
            embedding_service: Service for generating embeddings
            vector_storage: Storage for vectors and documents
        """
        self.embedding_service = embedding_service
        self.vector_storage = vector_storage
        logger.info("VectorIndexer initialized")

    def index_documents(
        self, documents: List[Dict[str, Any]], ner_re_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Generate embeddings and store in vector database.

        This method takes the documents and NER/RE results from KnowledgeGraphConstructor,
        prepares them for embedding, generates embeddings, and stores them in the vector database.

        Args:
            documents: List of document chunks with id, text, and metadata
            ner_re_results: Dictionary mapping document IDs to extracted entities and relations
        """
        logger.info(
            f"Starting embedding generation and storage for {len(documents)} documents"
        )

        # Prepare documents for embedding with metadata
        embedding_docs = self._prepare_documents_for_embedding(
            documents, ner_re_results
        )

        if embedding_docs:
            # Generate embeddings
            processed_docs = self.embedding_service.process_documents(embedding_docs)

            if processed_docs:
                # Store in vector database
                try:
                    # Initialize table with first batch if needed
                    if not self.vector_storage.table:
                        self.vector_storage.create_table(processed_docs[:1])

                    # Add all documents
                    self.vector_storage.add_documents(processed_docs)
                    logger.info(
                        f"Stored {len(processed_docs)} document embeddings in vector storage"
                    )
                except Exception as e:
                    logger.error(f"Failed to store embeddings: {e}")
                    raise
            else:
                logger.warning("Failed to generate embeddings")
        else:
            logger.warning("No documents prepared for embedding")

    def _prepare_documents_for_embedding(
        self, documents: List[Dict[str, Any]], ner_re_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for embedding by adding NER/RE metadata.

        Args:
            documents: Original documents
            ner_re_results: Results from NER/RE processing

        Returns:
            List of documents with metadata for embedding
        """
        embedding_docs = []

        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")

            if not text or doc_id not in ner_re_results:
                continue

            # Get NER/RE results
            doc_results = ner_re_results[doc_id]
            entities = doc_results.get("entities", [])
            triples = doc_results.get("triples", [])

            # Extract company name from title or entities
            company_name = doc.get("title", "")
            if not company_name and entities:
                # Try to find first COMPANY entity
                for entity in entities:
                    if entity.get("type") == "COMPANY":
                        company_name = entity["text"]
                        break

            # Create document for embedding
            embedding_doc = {
                "text": text,
                "doc_id": doc_id,
                "chunk_index": 0,  # Single chunk for now
                "company_name": company_name or "Unknown",
                "entities": entities,  # Already in typed format from Story 1.2.1
                "relations": triples,
                "source_file": doc.get("source_file", "corpus.json"),
            }

            embedding_docs.append(embedding_doc)

        return embedding_docs
