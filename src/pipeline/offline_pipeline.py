"""
Offline data processing pipeline orchestration.
"""

import logging
from pathlib import Path
from typing import Optional
import json
import pickle

from src.components.data_ingestor import DataIngestor
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.components.vector_indexer import VectorIndexer
from src.adapters.deepseek_adapter import DeepSeekAdapter
from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage
from config.settings import Settings

logger = logging.getLogger(__name__)


def run_offline_pipeline(
    corpus_path: str = "data/corpus.json",
    output_dir: str = "output",
    resume_from_checkpoint: bool = False,
) -> None:
    """
    Orchestrate the complete offline data processing pipeline.

    Args:
        corpus_path: Path to the corpus JSON file
        output_dir: Directory to store output files
        resume_from_checkpoint: Whether to resume from a previous checkpoint

    Raises:
        FileNotFoundError: If corpus file doesn't exist
        Exception: For any component initialization or processing failures
    """
    logger.info(f"Starting offline pipeline with corpus: {corpus_path}")

    # Validate corpus path
    corpus_path_obj = Path(corpus_path)
    if not corpus_path_obj.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Create output directories
    output_path = Path(output_dir)
    graph_dir = output_path / "graph"
    vector_dir = output_path / "vector_store"

    graph_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Initialize all components with proper dependencies
        logger.info("Initializing pipeline components...")

        # Initialize adapters and services
        settings = Settings()
        llm_adapter = DeepSeekAdapter()
        embedding_service = EmbeddingService()
        vector_storage = VectorStorage(
            db_path=vector_dir, table_name="a_share_rag"
        )

        # Initialize components
        data_ingestor = DataIngestor()
        knowledge_graph_constructor = KnowledgeGraphConstructor(llm_adapter=llm_adapter)
        vector_indexer = VectorIndexer(
            embedding_service=embedding_service, vector_storage=vector_storage
        )

        logger.info("All components initialized successfully")

        # Check for checkpoint if resume is requested
        checkpoint_file = output_path / "checkpoint.json"
        skip_graph_construction = False
        ner_re_results = None

        if resume_from_checkpoint and checkpoint_file.exists():
            logger.info("Loading checkpoint...")
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            if checkpoint.get("graph_construction_completed", False):
                skip_graph_construction = True
                # Load NER/RE results from checkpoint
                ner_re_file = graph_dir / "ner_re_results.json"
                if ner_re_file.exists():
                    with open(ner_re_file, "r", encoding="utf-8") as f:
                        ner_re_results = json.load(f)
                    logger.info("Loaded NER/RE results from checkpoint")

        # Step 2: Load and preprocess documents
        logger.info("Loading and preprocessing documents...")
        document_objs = data_ingestor.load_corpus(str(corpus_path))
        logger.info(f"Loaded {len(document_objs)} documents")
        
        # Convert Document objects to dictionaries for compatibility
        documents = []
        for doc in document_objs:
            documents.append({
                "id": f"doc_{doc.idx}",
                "text": doc.text,
                "title": doc.title,
                "idx": doc.idx
            })
        logger.info(f"Converted {len(documents)} documents to dictionary format")

        # Step 3: Process with KnowledgeGraphConstructor (if not skipped)
        if not skip_graph_construction:
            logger.info("Constructing knowledge graph...")
            ner_re_results, knowledge_graph = (
                knowledge_graph_constructor.process_documents(documents)
            )

            # Save graph and results
            logger.info("Saving knowledge graph...")
            with open(graph_dir / "graph.pkl", "wb") as f:
                pickle.dump(knowledge_graph, f)

            # Save NER/RE results for checkpoint
            with open(graph_dir / "ner_re_results.json", "w", encoding="utf-8") as f:
                json.dump(ner_re_results, f, ensure_ascii=False, indent=2)

            # Extract and save metadata from nested structure
            all_entities = []
            all_relations = []
            for doc_id, doc_results in ner_re_results.items():
                all_entities.extend(doc_results.get("entities", []))
                all_relations.extend(doc_results.get("triples", []))
            
            entity_metadata = {
                "total_entities": len(all_entities),
                "entity_types": list(
                    set(
                        e.get("type", "UNKNOWN")
                        for e in all_entities
                    )
                ),
            }
            relation_metadata = {
                "total_relations": len(all_relations),
                "relation_types": list(
                    set(
                        r[1] if isinstance(r, list) and len(r) > 1 else "UNKNOWN"
                        for r in all_relations
                    )
                ),
            }

            with open(graph_dir / "entity_metadata.json", "w", encoding="utf-8") as f:
                json.dump(entity_metadata, f, ensure_ascii=False, indent=2)

            with open(graph_dir / "relation_metadata.json", "w", encoding="utf-8") as f:
                json.dump(relation_metadata, f, ensure_ascii=False, indent=2)

            # Update checkpoint
            checkpoint = {"graph_construction_completed": True}
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)

            logger.info("Knowledge graph construction completed")

        # Step 4: Index with VectorIndexer
        logger.info("Indexing documents with vector embeddings...")
        vector_indexer.index_documents(documents, ner_re_results)
        logger.info("Vector indexing completed")

        # Clear checkpoint on successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        logger.info("Offline pipeline completed successfully")

    except ImportError as e:
        logger.error(f"Failed to import required component: {e}")
        raise ImportError(f"Component import failed: {e}") from e
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise Exception(f"Pipeline execution failed: {e}") from e
