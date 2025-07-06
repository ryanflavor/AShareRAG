"""Knowledge Graph Constructor component for NER and RE orchestration."""

import gc
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import igraph as ig
import psutil

from config.settings import Settings
from src.adapters import LLMAdapter
from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


class KnowledgeGraphConstructor:
    """Orchestrates Named Entity Recognition and Relation Extraction for building knowledge graphs."""

    # Entity type priority (higher number = more specific)
    ENTITY_TYPE_PRIORITY: ClassVar[dict[str, int]] = {
        "UNKNOWN": 0,
        "COMPANY": 1,
        "SUBSIDIARY": 2,
        "LISTED_COMPANY": 3,
        "COMPANY_CODE": 1,
        "PERSON": 1,
        "EXECUTIVE": 2,
        "TECHNOLOGY": 1,
        "PRODUCT": 1,
        "INDUSTRY": 1,
    }

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_storage: VectorStorage | None = None,
    ):
        """
        Initialize Knowledge Graph Constructor with optional embedding components.

        Args:
            embedding_service: Optional embedding service for text vectorization
            vector_storage: Optional vector storage for persisting embeddings
        """
        self.llm_adapter = LLMAdapter(enable_cache=True, high_throughput=True)
        self.graph = None
        self.embedding_service = embedding_service
        self.vector_storage = vector_storage
        self.deduplication_stats = {
            "total_entities": 0,
            "unique_entities": 0,
            "merged_entities": 0,
            "total_relations": 0,
            "unique_relations": 0,
            "merged_relations": 0,
        }

    def process_documents(
        self, documents: list[dict], batch_size: int | None = None
    ) -> tuple[dict, ig.Graph]:
        """
        Process documents to extract named entities and relations, building a knowledge graph.

        Args:
            documents: List of documents with 'id' and 'text' fields
            batch_size: Number of documents to process per batch (default: None - process all at once)

        Returns:
            Tuple of:
                - Dictionary mapping document IDs to extracted data (entities and triples)
                - igraph Graph object containing the knowledge graph

        Example:
            >>> constructor = KnowledgeGraphConstructor()
            >>> docs = [{"id": "doc1", "text": "综艺股份(600770)是一家科技公司"}]
            >>> results, graph = constructor.process_documents(docs)
            >>> # results["doc1"]["entities"] = [{"text": "综艺股份", "type": "COMPANY"}, ...]
            >>> # results["doc1"]["triples"] = [["综艺股份", "公司代码是", "600770"]]
            >>> # graph contains vertices and edges from the triples
        """
        if not documents:
            logger.info("No documents to process")
            return {}, ig.Graph(directed=True)

        logger.info(f"Starting NER and RE processing for {len(documents)} documents")

        # Initialize results and graph
        results = {}
        self.graph = ig.Graph(directed=True)

        # Track unique vertices and edges
        vertex_map = {}  # entity_text -> vertex_id
        edge_map = {}  # (subject_id, object_id, relation) -> edge_id

        # Reset deduplication stats
        self.deduplication_stats = {
            "total_entities": 0,
            "unique_entities": 0,
            "merged_entities": 0,
            "total_relations": 0,
            "unique_relations": 0,
            "merged_relations": 0,
        }

        # Use batch processing if batch_size is specified
        if batch_size and batch_size > 0:
            return self._process_documents_in_batches(
                documents, batch_size, results, vertex_map, edge_map
            )

        # Process all documents at once
        self._check_memory_usage()  # Check memory before processing
        self._process_document_batch(documents, results, vertex_map, edge_map)

        # Finalize processing
        self._finalize_processing(documents, results)

        # Step 5: Generate and store embeddings if services are configured
        if self.embedding_service and self.vector_storage:
            logger.info("Starting embedding generation and storage")

            # Prepare documents for embedding with metadata
            embedding_docs = self._prepare_documents_for_embedding(documents, results)

            if embedding_docs:
                # Generate embeddings
                processed_docs = self.embedding_service.process_documents(
                    embedding_docs
                )

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
                else:
                    logger.warning("Failed to generate embeddings")
            else:
                logger.warning("No documents prepared for embedding")

        return results, self.graph

    def _prepare_documents_for_embedding(
        self, documents: list[dict[str, Any]], ner_re_results: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
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

    def save_graph(self, file_path: Path | None = None, max_retries: int = 3) -> bool:
        """
        Save the graph to a GraphML file with comprehensive error handling.

        Args:
            file_path: Path where to save the graph. If None, uses default from settings
            max_retries: Maximum number of retry attempts for transient failures

        Returns:
            True if save successful, False otherwise
        """
        if not self.graph:
            logger.error("No graph to save")
            return False

        # Use default path if not provided
        if file_path is None:
            settings = Settings()
            file_path = settings.graph_storage_path / "knowledge_graph.graphml"

        # Ensure directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle backup rotation
        self._rotate_backups(file_path)

        # Check file permissions
        if not os.access(file_path.parent, os.W_OK):
            logger.error(f"No write permission for directory: {file_path.parent}")
            return False

        # Check disk space (require at least 10MB free)
        required_space = 10 * 1024 * 1024  # 10MB in bytes
        try:
            disk_usage = shutil.disk_usage(file_path.parent)
            if disk_usage.free < required_space:
                logger.error(
                    f"Insufficient disk space. Required: ~10MB, "
                    f"Available: {disk_usage.free / (1024 * 1024):.2f}MB"
                )
                return False
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        # Try to save with retry logic
        for attempt in range(max_retries):
            try:
                # Save graph
                logger.info(f"Saving graph to {file_path}")
                self.graph.write(str(file_path), format="graphml")

                # Verify file was created and has content
                if not file_path.exists() or file_path.stat().st_size == 0:
                    raise OSError("Graph file was not created or is empty")

                logger.info(f"Successfully saved graph to {file_path}")

                # Save metadata
                self._save_graph_metadata(file_path)

                return True

            except OSError as e:
                logger.error(
                    f"Failed to save graph (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts failed")

            except Exception as e:
                # Non-retriable errors
                logger.error(f"Non-retriable error while saving graph: {e}")
                return False

        return False

    def load_graph(self, file_path: Path) -> ig.Graph | None:
        """
        Load a graph from a GraphML file with error handling.

        Args:
            file_path: Path to the GraphML file

        Returns:
            Loaded graph or None if loading failed
        """
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            logger.error(f"Graph file not found: {file_path}")
            return None

        # Check read permissions
        if not os.access(file_path, os.R_OK):
            logger.error(f"No read permission for file: {file_path}")
            return None

        # Check file size (max 2GB)
        max_size = 2 * 1024 * 1024 * 1024  # 2GB in bytes
        file_size = file_path.stat().st_size
        if file_size > max_size:
            logger.error(
                f"Graph file too large: {file_size / (1024 * 1024):.2f}MB "
                f"(max: {max_size / (1024 * 1024):.2f}MB)"
            )
            return None

        try:
            logger.info(f"Loading graph from {file_path}")

            # Load the graph
            graph = ig.read(str(file_path), format="graphml")

            # Basic validation
            if graph.vcount() == 0:
                logger.warning("Loaded graph has no vertices")

            logger.info(
                f"Successfully loaded graph with {graph.vcount()} vertices "
                f"and {graph.ecount()} edges"
            )

            self.graph = graph

            # Load metadata if available
            self._load_graph_metadata(file_path)

            return graph

        except Exception as e:
            logger.error(f"Failed to load graph from {file_path}: {e}")
            return None

    def _save_graph_metadata(self, graph_file_path: Path) -> None:
        """Save graph metadata to a JSON file."""
        metadata_path = graph_file_path.parent / f"{graph_file_path.stem}_metadata.json"

        try:
            metadata = {
                "creation_timestamp": datetime.now(timezone.utc).isoformat(),
                "graph_file": graph_file_path.name,
                "vertices_count": self.graph.vcount(),
                "edges_count": self.graph.ecount(),
                "is_directed": self.graph.is_directed(),
                "connected_components": len(self.graph.components())
                if not self.graph.is_directed()
                else "N/A",
                "format_version": "1.0",
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved graph metadata to {metadata_path}")

        except Exception as e:
            logger.warning(f"Failed to save graph metadata: {e}")

    def _load_graph_metadata(self, graph_file_path: Path) -> dict | None:
        """Load graph metadata from JSON file."""
        metadata_path = graph_file_path.parent / f"{graph_file_path.stem}_metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            logger.info(f"Loaded graph metadata from {metadata_path}")
            return metadata

        except Exception as e:
            logger.warning(f"Failed to load graph metadata: {e}")
            return None

    def _rotate_backups(self, file_path: Path, max_backups: int = 3) -> None:
        """
        Rotate backup files to maintain only max_backups number of backups.

        Args:
            file_path: Main file path
            max_backups: Maximum number of backup files to keep
        """
        try:
            if not file_path.exists():
                return
            # Create backup filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"graph_backup_{timestamp}.graphml"
            backup_path = file_path.parent / backup_name

            # Copy current file to backup
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

            # Also backup metadata if exists
            metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
            if metadata_path.exists():
                metadata_backup = (
                    file_path.parent / f"graph_backup_{timestamp}_metadata.json"
                )
                shutil.copy2(metadata_path, metadata_backup)

            # Remove old backups
            backup_files = sorted(
                file_path.parent.glob("graph_backup_*.graphml"),
                key=lambda p: p.stat().st_mtime,
            )

            if len(backup_files) > max_backups:
                for old_backup in backup_files[:-max_backups]:
                    old_backup.unlink()
                    logger.info(f"Removed old backup: {old_backup}")

                    # Remove associated metadata
                    old_metadata = (
                        old_backup.parent / f"{old_backup.stem}_metadata.json"
                    )
                    if old_metadata.exists():
                        old_metadata.unlink()

        except Exception as e:
            logger.warning(f"Failed to rotate backups: {e}")

    def calculate_graph_statistics(self) -> dict[str, Any]:
        """
        Calculate comprehensive statistics about the graph.

        Returns:
            Dictionary containing graph statistics
        """
        if not self.graph or self.graph.vcount() == 0:
            return {
                "vertices_count": 0,
                "edges_count": 0,
                "density": 0.0,
                "average_degree": 0.0,
                "max_degree": 0,
                "degree_distribution": {},
                "entity_type_distribution": {},
                "connected_components": 0,
            }

        # Basic metrics
        vertices_count = self.graph.vcount()
        edges_count = self.graph.ecount()

        # Density (for directed graph)
        max_possible_edges = vertices_count * (vertices_count - 1)
        density = edges_count / max_possible_edges if max_possible_edges > 0 else 0.0

        # Degree statistics
        degrees = self.graph.degree(mode="all")
        average_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0

        # Degree distribution
        degree_distribution = {}
        for d in degrees:
            degree_distribution[d] = degree_distribution.get(d, 0) + 1

        # Entity type distribution
        entity_type_distribution = {}
        for vertex in self.graph.vs:
            entity_type = (
                vertex["entity_type"]
                if "entity_type" in vertex.attributes()
                else "UNKNOWN"
            )
            entity_type_distribution[entity_type] = (
                entity_type_distribution.get(entity_type, 0) + 1
            )

        # Connected components (for undirected version)
        components = self.graph.components(mode="weak")

        stats = {
            "vertices_count": vertices_count,
            "edges_count": edges_count,
            "density": round(density, 4),
            "average_degree": round(average_degree, 2),
            "max_degree": max_degree,
            "degree_distribution": degree_distribution,
            "entity_type_distribution": entity_type_distribution,
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components)
            if components
            else 0,
        }

        logger.info(
            f"Graph statistics: {vertices_count} vertices, {edges_count} edges, "
            f"density={stats['density']}, avg_degree={stats['average_degree']}, "
            f"components={stats['connected_components']}"
        )

        return stats

    def get_top_entities_by_degree(
        self, k: int = 10, mode: str = "all"
    ) -> list[dict[str, Any]]:
        """
        Get the top k entities with highest degree centrality.

        Args:
            k: Number of top entities to return
            mode: Degree mode - "all", "in", or "out"

        Returns:
            List of dictionaries with entity information sorted by degree
        """
        if not self.graph or self.graph.vcount() == 0:
            return []

        # Calculate degrees
        degrees = self.graph.degree(mode=mode)

        # Create list of (vertex_id, degree) tuples
        vertex_degrees = list(enumerate(degrees))

        # Sort by degree (descending)
        vertex_degrees.sort(key=lambda x: x[1], reverse=True)

        # Get top k
        top_k = vertex_degrees[:k]

        # Build result list
        results = []
        for vertex_id, degree in top_k:
            vertex = self.graph.vs[vertex_id]
            results.append(
                {
                    "name": vertex["name"],
                    "entity_type": vertex["entity_type"]
                    if "entity_type" in vertex.attributes()
                    else "UNKNOWN",
                    "degree": degree,
                    "occurrence_count": vertex["occurrence_count"]
                    if "occurrence_count" in vertex.attributes()
                    else 1,
                    "first_seen": vertex["first_seen"]
                    if "first_seen" in vertex.attributes()
                    else "unknown",
                }
            )

        logger.info(
            f"Top {len(results)} entities by degree: {[e['name'] for e in results[:5]]}"
        )

        return results

    def analyze_connected_components(self) -> dict[str, Any]:
        """
        Analyze the connected components in the graph.

        Returns:
            Dictionary with component analysis results
        """
        if not self.graph or self.graph.vcount() == 0:
            return {
                "total_components": 0,
                "largest_component_size": 0,
                "isolated_vertices": 0,
                "component_sizes": [],
            }

        # Get weakly connected components (treats directed edges as undirected)
        components = self.graph.components(mode="weak")

        # Component sizes
        component_sizes = [len(c) for c in components]
        component_sizes.sort(reverse=True)

        # Count isolated vertices (components of size 1)
        isolated_vertices = sum(1 for size in component_sizes if size == 1)

        results = {
            "total_components": len(components),
            "largest_component_size": component_sizes[0] if component_sizes else 0,
            "isolated_vertices": isolated_vertices,
            "component_sizes": component_sizes[:10],  # Top 10 component sizes
            "component_size_distribution": {
                "size_1": isolated_vertices,
                "size_2_10": sum(1 for s in component_sizes if 2 <= s <= 10),
                "size_11_100": sum(1 for s in component_sizes if 11 <= s <= 100),
                "size_100+": sum(1 for s in component_sizes if s > 100),
            },
        }

        logger.info(
            f"Connected components: {results['total_components']} total, "
            f"largest={results['largest_component_size']}, "
            f"isolated={results['isolated_vertices']}"
        )

        return results

    def _process_document_batch(
        self,
        documents: list[dict],
        results: dict,
        vertex_map: dict,
        edge_map: dict,
        start_idx: int = 0,
    ) -> None:
        """Process a batch of documents and update the graph."""
        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{start_idx + idx}")
            text = doc.get("text", "")

            logger.info(f"Processing document {start_idx + idx + 1}: {doc_id}")

            # Step 1: Extract entities using LLM adapter
            entities = self.llm_adapter.extract_entities(text)
            logger.info(f"Extracted {len(entities)} entities from document {doc_id}")
            self.deduplication_stats["total_entities"] += len(entities)

            # Step 2: Extract relations using entities
            triples = []
            if entities:
                triples = self.llm_adapter.extract_relations(text, entities)
                logger.info(f"Extracted {len(triples)} triples from document {doc_id}")
                self.deduplication_stats["total_relations"] += len(triples)

            # Store results
            results[doc_id] = {"entities": entities, "triples": triples}

            # Step 3: Add entities to graph as vertices with enhanced deduplication
            for entity in entities:
                entity_text = entity["text"]
                entity_type = entity["type"]

                if entity_text not in vertex_map:
                    # Add new vertex
                    vertex_id = self.graph.vcount()
                    vertex_map[entity_text] = vertex_id
                    self.graph.add_vertex(
                        name=entity_text,
                        entity_type=entity_type,
                        first_seen=doc_id,
                        occurrence_count=1,
                        source_docs=[doc_id],
                    )
                    self.deduplication_stats["unique_entities"] += 1
                else:
                    # Update existing vertex
                    vertex_id = vertex_map[entity_text]
                    vertex = self.graph.vs[vertex_id]

                    # Update occurrence count
                    vertex["occurrence_count"] += 1

                    # Update source docs
                    if doc_id not in vertex["source_docs"]:
                        vertex["source_docs"].append(doc_id)

                    # Update entity type if more specific
                    current_priority = self.ENTITY_TYPE_PRIORITY.get(
                        vertex["entity_type"], 0
                    )
                    new_priority = self.ENTITY_TYPE_PRIORITY.get(entity_type, 0)

                    if new_priority > current_priority:
                        logger.debug(
                            f"Updating entity type for '{entity_text}' "
                            f"from {vertex['entity_type']} to {entity_type}"
                        )
                        vertex["entity_type"] = entity_type

                    self.deduplication_stats["merged_entities"] += 1

            # Step 4: Add triples to graph as edges with enhanced merging
            for triple in triples:
                subject, predicate, obj = triple

                # Ensure both subject and object are in the graph
                for entity_text in [subject, obj]:
                    if entity_text not in vertex_map:
                        # Add vertex even if not in entity list (for completeness)
                        vertex_id = self.graph.vcount()
                        vertex_map[entity_text] = vertex_id
                        self.graph.add_vertex(
                            name=entity_text,
                            entity_type="UNKNOWN",  # Not in NER results
                            first_seen=doc_id,
                            occurrence_count=1,
                            source_docs=[doc_id],
                        )
                        self.deduplication_stats["unique_entities"] += 1

                # Get vertex IDs
                subject_id = vertex_map[subject]
                object_id = vertex_map[obj]

                # Check for self-referential relations
                if subject_id == object_id:
                    logger.debug(
                        f"Self-referential relation found: {subject} -> {predicate} -> {obj}"
                    )

                # Check if edge already exists
                edge_key = (subject_id, object_id, predicate)

                if edge_key not in edge_map:
                    # Add new edge
                    edge_id = self.graph.ecount()
                    self.graph.add_edge(
                        subject_id,
                        object_id,
                        relation=predicate,
                        source_docs=[doc_id],
                        confidence=1.0,
                        first_seen=doc_id,
                    )
                    edge_map[edge_key] = edge_id
                    self.deduplication_stats["unique_relations"] += 1
                else:
                    # Update existing edge
                    edge_id = edge_map[edge_key]
                    edge = self.graph.es[edge_id]

                    # Update source docs
                    if doc_id not in edge["source_docs"]:
                        edge["source_docs"].append(doc_id)

                    self.deduplication_stats["merged_relations"] += 1

    def _process_documents_in_batches(
        self,
        documents: list[dict],
        batch_size: int,
        results: dict,
        vertex_map: dict,
        edge_map: dict,
    ) -> tuple[dict, ig.Graph]:
        """Process documents in batches with memory cleanup."""
        total_docs = len(documents)
        num_batches = (total_docs + batch_size - 1) // batch_size

        logger.info(
            f"Processing {total_docs} documents in {num_batches} batches of {batch_size}"
        )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_docs)
            batch_docs = documents[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{num_batches} (docs {start_idx + 1}-{end_idx})"
            )

            # Check memory before processing batch
            self._check_memory_usage()

            # Process the batch
            self._process_document_batch(
                batch_docs, results, vertex_map, edge_map, start_idx
            )

            # Clean up memory after batch
            if batch_idx < num_batches - 1:  # Don't clean up after last batch
                gc.collect()
                logger.info("Memory cleanup performed after batch processing")

            # Check if we should prune the graph
            if self._should_prune_graph():
                self._prune_graph()

        # Final statistics and cleanup
        self._finalize_processing(documents, results)

        return results, self.graph

    def _check_memory_usage(self) -> None:
        """Check and log memory usage."""
        try:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent

            if memory_percent > 80:
                logger.warning(
                    f"High memory usage detected: {memory_percent:.1f}% "
                    f"({memory_info.used / (1024**3):.1f}GB used of "
                    f"{memory_info.total / (1024**3):.1f}GB total)"
                )
            else:
                logger.debug(f"Memory usage: {memory_percent:.1f}%")
        except Exception as e:
            logger.debug(f"Could not check memory usage: {e}")

    def _should_prune_graph(self) -> bool:
        """Determine if graph should be pruned based on size."""
        if not self.graph:
            return False

        # Prune if graph has more than 1M edges
        return self.graph.ecount() > 1_000_000

    def _prune_graph(self) -> None:
        """Prune graph by removing low-degree vertices."""
        if not self.graph or self.graph.vcount() == 0:
            return

        logger.info("Pruning graph to reduce memory usage")

        # Get vertices with degree < 2
        degrees = self.graph.degree()
        vertices_to_remove = [i for i, d in enumerate(degrees) if d < 2]

        if vertices_to_remove:
            logger.info(f"Removing {len(vertices_to_remove)} low-degree vertices")
            self.graph.delete_vertices(vertices_to_remove)

            # Note: vertex_map will be invalid after this, but it's only used during construction
            logger.warning("Graph pruned - vertex mappings may be invalid")

    def _finalize_processing(self, documents: list[dict], results: dict) -> None:
        """Finalize processing with statistics and logging."""
        logger.info(
            f"Completed NER and RE processing. "
            f"Total documents: {len(documents)}, "
            f"Total vertices: {self.graph.vcount()}, "
            f"Total edges: {self.graph.ecount()}"
        )

        # Log deduplication statistics
        logger.info(
            f"Deduplication statistics: "
            f"Total entities: {self.deduplication_stats['total_entities']}, "
            f"Unique entities: {self.deduplication_stats['unique_entities']}, "
            f"Merged entities: {self.deduplication_stats['merged_entities']}, "
            f"Total relations: {self.deduplication_stats['total_relations']}, "
            f"Unique relations: {self.deduplication_stats['unique_relations']}, "
            f"Merged relations: {self.deduplication_stats['merged_relations']}"
        )

        # Calculate and log graph statistics
        self.calculate_graph_statistics()

        # Log top entities
        top_entities = self.get_top_entities_by_degree(k=5)
        if top_entities:
            logger.info(f"Top entities by degree: {[e['name'] for e in top_entities]}")
