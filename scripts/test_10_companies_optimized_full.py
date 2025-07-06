#!/usr/bin/env python3
"""
Full-stack optimized pipeline for knowledge graph construction with:
1. Parallel NER/RE extraction (LLM API calls)
2. Batch embedding generation 
3. Optimized graph construction
4. Vector storage integration
"""

import json
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FullOptimizedKnowledgeGraphConstructor(KnowledgeGraphConstructor):
    """
    Fully optimized knowledge graph constructor with:
    - Parallel NER/RE processing
    - Optimized embedding generation
    - Efficient graph construction
    """
    
    def __init__(
        self, 
        max_workers: int = 5,
        enable_embeddings: bool = True,
        embedding_batch_size: int = 10
    ):
        """
        Initialize with full optimization capabilities.
        
        Args:
            max_workers: Number of parallel LLM API workers
            enable_embeddings: Whether to generate embeddings
            embedding_batch_size: Batch size for embedding generation
        """
        # Initialize embedding services if enabled
        embedding_service = None
        vector_storage = None
        
        if enable_embeddings:
            embedding_service = EmbeddingService(batch_size=embedding_batch_size)
            vector_storage = VectorStorage()
            # Connect to vector database
            try:
                vector_storage.connect()
                logger.info("✅ Vector storage connected successfully")
            except Exception as e:
                logger.error(f"❌ Failed to connect vector storage: {e}")
                vector_storage = None
        
        super().__init__(embedding_service=embedding_service, vector_storage=vector_storage)
        
        self.max_workers = max_workers
        self.enable_embeddings = enable_embeddings
        self.embedding_batch_size = embedding_batch_size
        
        logger.info(f"🚀 FullOptimized initialized:")
        logger.info(f"   • LLM workers: {max_workers}")
        logger.info(f"   • Embeddings: {'✅' if enable_embeddings else '❌'}")
        logger.info(f"   • Embedding batch size: {embedding_batch_size}")
    
    def process_documents_full_optimized(self, documents: List[Dict]) -> Tuple[Dict, Any]:
        """
        Execute the full optimized three-stage pipeline.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Tuple of (processing_results, graph)
        """
        logger.info("="*60)
        logger.info("🚀 FULL-STACK OPTIMIZED PIPELINE")
        logger.info("="*60)
        
        total_start = time.time()
        
        # Stage 1: Parallel NER/RE Extraction
        logger.info("\n📡 STAGE 1: Parallel NER/RE Extraction")
        logger.info("-" * 40)
        stage1_start = time.time()
        
        extraction_results = self._parallel_extract_entities_and_relations(documents)
        
        stage1_time = time.time() - stage1_start
        logger.info(f"⚡ Stage 1 completed in {stage1_time:.2f}s")
        
        # Stage 2: Knowledge Graph Construction
        logger.info("\n🔗 STAGE 2: Knowledge Graph Construction")
        logger.info("-" * 40)
        stage2_start = time.time()
        
        results, graph = self._build_graph_from_extractions(documents, extraction_results)
        
        stage2_time = time.time() - stage2_start
        logger.info(f"📊 Stage 2 completed in {stage2_time:.2f}s")
        
        # Stage 3: Embedding Generation & Vector Storage (if enabled)
        stage3_time = 0
        if self.enable_embeddings and self.embedding_service:
            logger.info("\n🎯 STAGE 3: Embedding Generation & Vector Storage")
            logger.info("-" * 40)
            stage3_start = time.time()
            
            self._process_embeddings_optimized(documents, results)
            
            stage3_time = time.time() - stage3_start
            logger.info(f"🎯 Stage 3 completed in {stage3_time:.2f}s")
        
        # Summary
        total_time = time.time() - total_start
        logger.info("\n" + "="*50)
        logger.info("⏱️  PERFORMANCE SUMMARY")
        logger.info("="*50)
        logger.info(f"Stage 1 (NER/RE):      {stage1_time:6.2f}s ({stage1_time/total_time*100:.1f}%)")
        logger.info(f"Stage 2 (Graph):       {stage2_time:6.2f}s ({stage2_time/total_time*100:.1f}%)")
        if stage3_time > 0:
            logger.info(f"Stage 3 (Embeddings):  {stage3_time:6.2f}s ({stage3_time/total_time*100:.1f}%)")
        logger.info(f"{'='*20}")
        logger.info(f"Total Time:            {total_time:6.2f}s")
        logger.info(f"Average per document:  {total_time/len(documents):6.2f}s")
        
        return results, graph
    
    def _parallel_extract_entities_and_relations(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract entities and relations in parallel with progress tracking.
        """
        results = [None] * len(documents)
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all extraction tasks
            future_to_index = {
                executor.submit(self._extract_single_document, doc): i
                for i, doc in enumerate(documents)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    doc_result = future.result()
                    results[index] = doc_result
                    completed_count += 1
                    
                    # Progress logging
                    doc_id = documents[index].get('id', f'doc_{index}')
                    title = documents[index].get('title', 'Unknown')
                    entities_count = len(doc_result.get('entities', []))
                    relations_count = len(doc_result.get('triples', []))
                    
                    logger.info(f"✅ [{completed_count:2d}/{len(documents)}] {title[:15]:<15} | "
                              f"E: {entities_count:2d} | R: {relations_count:2d}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process document {index}: {e}")
                    results[index] = {'entities': [], 'triples': []}
                    completed_count += 1
        
        return results
    
    def _extract_single_document(self, doc: Dict) -> Dict:
        """Extract entities and relations from a single document."""
        text = doc.get('text', '')
        
        # NER extraction
        entities = self.llm_adapter.extract_entities(text)
        
        # RE extraction
        triples = []
        if entities:
            triples = self.llm_adapter.extract_relations(text, entities)
        
        return {
            'entities': entities,
            'triples': triples
        }
    
    def _build_graph_from_extractions(self, documents: List[Dict], extraction_results: List[Dict]) -> Tuple[Dict, Any]:
        """
        Build knowledge graph from pre-extracted data with optimization.
        """
        import igraph as ig
        
        # Initialize
        self.graph = ig.Graph(directed=True)
        vertex_map = {}  # entity_text -> vertex_id
        edge_map = {}    # (subject_id, object_id, relation) -> edge_id
        results = {}
        
        # Reset stats
        self.deduplication_stats = {
            "total_entities": 0,
            "unique_entities": 0,
            "merged_entities": 0,
            "total_relations": 0,
            "unique_relations": 0,
            "merged_relations": 0,
        }
        
        # Process each document
        for i, (doc, extractions) in enumerate(zip(documents, extraction_results)):
            doc_id = doc.get('id', f'doc_{i}')
            title = doc.get('title', 'Unknown')
            entities = extractions.get('entities', [])
            triples = extractions.get('triples', [])
            
            # Store results
            results[doc_id] = {
                'entities': entities,
                'triples': triples
            }
            
            # Update stats
            self.deduplication_stats["total_entities"] += len(entities)
            self.deduplication_stats["total_relations"] += len(triples)
            
            # Add entities as vertices
            for entity in entities:
                entity_text = entity["text"]
                entity_type = entity["type"]
                
                if entity_text not in vertex_map:
                    # New vertex
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
                    vertex["occurrence_count"] += 1
                    
                    if doc_id not in vertex["source_docs"]:
                        vertex["source_docs"].append(doc_id)
                    
                    # Update entity type if more specific
                    current_priority = self.ENTITY_TYPE_PRIORITY.get(vertex["entity_type"], 0)
                    new_priority = self.ENTITY_TYPE_PRIORITY.get(entity_type, 0)
                    
                    if new_priority > current_priority:
                        vertex["entity_type"] = entity_type
                    
                    self.deduplication_stats["merged_entities"] += 1
            
            # Add relations as edges
            for triple in triples:
                subject, predicate, obj = triple
                
                # Ensure entities exist
                for entity_text in [subject, obj]:
                    if entity_text not in vertex_map:
                        vertex_id = self.graph.vcount()
                        vertex_map[entity_text] = vertex_id
                        self.graph.add_vertex(
                            name=entity_text,
                            entity_type="UNKNOWN",
                            first_seen=doc_id,
                            occurrence_count=1,
                            source_docs=[doc_id],
                        )
                        self.deduplication_stats["unique_entities"] += 1
                
                # Add edge
                subject_id = vertex_map[subject]
                object_id = vertex_map[obj]
                edge_key = (subject_id, object_id, predicate)
                
                if edge_key not in edge_map:
                    # New edge
                    self.graph.add_edge(
                        subject_id,
                        object_id,
                        relation=predicate,
                        source_docs=[doc_id],
                        confidence=1.0,
                        first_seen=doc_id,
                    )
                    edge_map[edge_key] = self.graph.ecount() - 1
                    self.deduplication_stats["unique_relations"] += 1
                else:
                    # Update existing edge
                    edge_id = edge_map[edge_key]
                    edge = self.graph.es[edge_id]
                    
                    if doc_id not in edge["source_docs"]:
                        edge["source_docs"].append(doc_id)
                    
                    self.deduplication_stats["merged_relations"] += 1
            
            # Progress logging for graph construction
            if (i + 1) % 2 == 0 or i == len(documents) - 1:
                logger.info(f"🔗 [{i+1:2d}/{len(documents)}] {title[:15]:<15} | "
                          f"V: {self.graph.vcount():3d} | E: {self.graph.ecount():3d}")
        
        # Finalize
        self._finalize_processing(documents, results)
        
        return results, self.graph
    
    def _process_embeddings_optimized(self, documents: List[Dict], results: Dict) -> None:
        """
        Generate embeddings with optimized batching.
        """
        if not self.embedding_service or not self.vector_storage:
            logger.warning("Embedding services not configured")
            return
        
        try:
            # Load embedding model
            logger.info("🔄 Loading embedding model...")
            model_loaded = self.embedding_service.load_model()
            if not model_loaded:
                logger.error("❌ Failed to load embedding model")
                return
            
            # Prepare documents for embedding
            logger.info("📝 Preparing documents for embedding...")
            embedding_docs = self._prepare_documents_for_embedding(documents, results)
            
            if not embedding_docs:
                logger.warning("⚠️  No documents prepared for embedding")
                return
            
            logger.info(f"🎯 Processing {len(embedding_docs)} documents in batches of {self.embedding_batch_size}")
            
            # Generate embeddings in optimized batches
            processed_docs = self.embedding_service.process_documents(embedding_docs)
            
            if not processed_docs:
                logger.error("❌ Failed to generate embeddings")
                return
            
            # Store in vector database
            logger.info("💾 Storing embeddings in vector database...")
            
            try:
                # Initialize table if needed
                if not self.vector_storage.table:
                    logger.info("🏗️  Creating new vector table...")
                    self.vector_storage.create_table(processed_docs[:1])
                
                # Add all documents
                self.vector_storage.add_documents(processed_docs)
                
                logger.info(f"✅ Stored {len(processed_docs)} document embeddings successfully")
                
            except Exception as vector_error:
                logger.error(f"❌ Vector storage failed: {vector_error}")
                logger.info("📈 Continuing without vector storage...")
            
        except Exception as e:
            logger.error(f"❌ Embedding processing failed: {e}")


def load_corpus_data(corpus_path: Path, limit: int = 10) -> List[Dict]:
    """Load documents from corpus."""
    logger.info(f"📂 Loading corpus from {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    logger.info(f"📊 Total documents available: {len(corpus_data)}")
    
    documents = []
    for i, doc in enumerate(corpus_data[:limit]):
        formatted_doc = {
            'id': f"company_{i:03d}",
            'title': doc.get('title', f'Company {i}'),
            'text': doc.get('text', ''),
            'idx': doc.get('idx', i)
        }
        documents.append(formatted_doc)
    
    company_names = [doc['title'] for doc in documents]
    logger.info(f"🏢 Selected companies: {', '.join(company_names)}")
    
    return documents


def run_full_optimized_pipeline(
    documents: List[Dict], 
    max_workers: int = 5,
    enable_embeddings: bool = True,
    embedding_batch_size: int = 10
) -> Tuple[Dict, str]:
    """Run the complete optimized pipeline."""
    
    logger.info("🚀 Initializing Full-Stack Optimized Constructor...")
    constructor = FullOptimizedKnowledgeGraphConstructor(
        max_workers=max_workers,
        enable_embeddings=enable_embeddings,
        embedding_batch_size=embedding_batch_size
    )
    
    # Execute pipeline
    results, graph = constructor.process_documents_full_optimized(documents)
    
    # Generate comprehensive statistics
    logger.info("\n" + "="*50)
    logger.info("📊 COMPREHENSIVE GRAPH ANALYSIS")
    logger.info("="*50)
    
    stats = constructor.calculate_graph_statistics()
    for key, value in stats.items():
        if isinstance(value, dict) and len(value) <= 10:
            logger.info(f"{key}:")
            for subkey, subvalue in value.items():
                logger.info(f"  {subkey}: {subvalue}")
        else:
            logger.info(f"{key}: {value}")
    
    # Top entities analysis
    logger.info("\n🏆 TOP ENTITIES BY CONNECTIVITY")
    logger.info("-" * 30)
    top_entities = constructor.get_top_entities_by_degree(k=10)
    for i, entity in enumerate(top_entities, 1):
        logger.info(f"{i:2d}. {entity['name'][:20]:<20} | "
                   f"{entity['entity_type']:<12} | "
                   f"Degree: {entity['degree']:2d}")
    
    # Connected components analysis
    logger.info("\n🔗 CONNECTED COMPONENTS ANALYSIS")
    logger.info("-" * 30)
    component_analysis = constructor.analyze_connected_components()
    for key, value in component_analysis.items():
        logger.info(f"{key}: {value}")
    
    # Save graph
    logger.info("\n💾 SAVING OUTPUTS")
    logger.info("-" * 20)
    settings = Settings()
    graph_file = settings.graph_storage_path / "10_companies_full_optimized.graphml"
    
    success = constructor.save_graph(graph_file)
    if success:
        file_size_kb = graph_file.stat().st_size / 1024
        logger.info(f"✅ Graph saved: {graph_file}")
        logger.info(f"📏 File size: {file_size_kb:.2f} KB")
        
        # Verify reload
        test_constructor = FullOptimizedKnowledgeGraphConstructor(enable_embeddings=False)
        loaded_graph = test_constructor.load_graph(graph_file)
        if loaded_graph:
            logger.info(f"✅ Verification: Graph reloaded successfully")
        else:
            logger.error("❌ Verification failed: Could not reload graph")
    else:
        logger.error("❌ Failed to save graph")
        return results, ""
    
    return results, str(graph_file)


def print_final_summary(results: Dict, graph_file: str, total_time: float, config: Dict):
    """Print comprehensive final summary."""
    logger.info("\n" + "="*60)
    logger.info("🎯 PIPELINE EXECUTION SUMMARY")
    logger.info("="*60)
    
    total_entities = sum(len(r.get('entities', [])) for r in results.values())
    total_relations = sum(len(r.get('triples', [])) for r in results.values())
    
    logger.info("📊 PROCESSING METRICS:")
    logger.info(f"   • Documents processed: {len(results)}")
    logger.info(f"   • Total entities extracted: {total_entities}")
    logger.info(f"   • Total relations extracted: {total_relations}")
    logger.info(f"   • Total processing time: {total_time:.2f} seconds")
    logger.info(f"   • Average time per document: {total_time/len(results):.2f} seconds")
    
    logger.info(f"\n⚙️  CONFIGURATION:")
    logger.info(f"   • LLM API workers: {config['max_workers']}")
    logger.info(f"   • Embeddings enabled: {'✅' if config['enable_embeddings'] else '❌'}")
    logger.info(f"   • Embedding batch size: {config['embedding_batch_size']}")
    
    logger.info(f"\n📁 OUTPUTS:")
    logger.info(f"   • Knowledge graph: {graph_file}")
    logger.info(f"   • Format: GraphML (compatible with Gephi, Cytoscape)")
    logger.info(f"   • Vector embeddings: {'Stored in LanceDB' if config['enable_embeddings'] else 'Not generated'}")
    
    logger.info(f"\n🎉 FULL-STACK OPTIMIZATION COMPLETE!")


def main():
    """Main execution function with configuration options."""
    try:
        # Configuration
        config = {
            'max_workers': 5,           # Parallel LLM API calls
            'enable_embeddings': True,  # Generate and store embeddings
            'embedding_batch_size': 10, # Embedding batch size
            'document_limit': 10        # Number of companies to process
        }
        
        # Setup paths
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        corpus_path = project_root / "data" / "corpus.json"
        
        logger.info("🚀 FULL-STACK OPTIMIZED KNOWLEDGE GRAPH PIPELINE")
        logger.info("="*60)
        logger.info(f"📂 Project: {project_root}")
        logger.info(f"📄 Corpus: {corpus_path}")
        logger.info(f"⚙️  Config: {config}")
        
        # Verify corpus
        if not corpus_path.exists():
            logger.error(f"❌ Corpus file not found: {corpus_path}")
            return 1
        
        # Execute pipeline
        start_time = time.time()
        
        documents = load_corpus_data(corpus_path, limit=config['document_limit'])
        results, graph_file = run_full_optimized_pipeline(
            documents, 
            max_workers=config['max_workers'],
            enable_embeddings=config['enable_embeddings'],
            embedding_batch_size=config['embedding_batch_size']
        )
        
        total_time = time.time() - start_time
        
        # Final summary
        print_final_summary(results, graph_file, total_time, config)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)