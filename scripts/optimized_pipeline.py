#!/usr/bin/env python3
"""
Optimized Pipeline with Parallel NER/RE Processing and Batch Embedding
Separates NER/RE processing (slow, parallelizable) from embedding (fast, batch)
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.adapters.deepseek_adapter import DeepSeekAdapter
from src.components.embedding_service import EmbeddingService
from src.components.vector_indexer import VectorIndexer
from src.components.vector_storage import VectorStorage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallelNERProcessor:
    """Handles parallel NER/RE processing with intermediate file storage."""
    
    def __init__(self, max_workers: int = 4, output_dir: str = "output/optimized_pipeline"):
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = DeepSeekAdapter(enable_cache=False, high_throughput=True)
        
    def process_single_document(self, doc_data: Tuple[str, Dict]) -> Tuple[str, Dict]:
        """Process a single document for NER/RE."""
        doc_id, company_data = doc_data
        
        try:
            logger.info(f"Processing {doc_id}: {company_data['title']}")
            text = company_data['text']
            
            # Extract entities
            start_time = time.time()
            entities = self.adapter.extract_entities(text, include_types=True)
            ner_time = time.time() - start_time
            
            # Extract relations
            relations = []
            re_time = 0
            if entities:
                start_time = time.time()
                relations = self.adapter.extract_relations(text, entities)
                re_time = time.time() - start_time
            
            total_time = ner_time + re_time
            
            result = {
                "doc_id": doc_id,
                "company_title": company_data['title'],
                "company_idx": company_data['idx'],
                "entities": entities,
                "triples": relations,
                "processing_time": {
                    "ner_time": ner_time,
                    "re_time": re_time,
                    "total_time": total_time
                },
                "text_length": len(text),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Completed {doc_id}: {len(entities)} entities, {len(relations)} relations in {total_time:.2f}s")
            return doc_id, result
            
        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            return doc_id, {
                "doc_id": doc_id,
                "error": str(e),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def process_documents_parallel(self, companies: List[Dict], resume: bool = True) -> Dict:
        """Process multiple documents in parallel."""
        # Prepare document data
        doc_data_list = []
        for i, company in enumerate(companies):
            doc_id = f"doc_{i}"
            doc_data_list.append((doc_id, company))
        
        # Check for existing results if resume is enabled
        results_file = self.output_dir / "ner_re_results.json"
        existing_results = {}
        processed_docs = set()
        
        if resume and results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
                processed_docs = set(existing_results.keys())
                logger.info(f"Resume mode: Found {len(processed_docs)} existing results")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        
        # Filter out already processed documents
        remaining_docs = [(doc_id, data) for doc_id, data in doc_data_list if doc_id not in processed_docs]
        
        if not remaining_docs:
            logger.info("All documents already processed")
            return existing_results
        
        logger.info(f"Processing {len(remaining_docs)} documents with {self.max_workers} workers")
        
        # Process in parallel
        start_time = time.time()
        results = existing_results.copy()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_document, doc_data) for doc_data in remaining_docs]
            
            for i, future in enumerate(futures):
                try:
                    doc_id, result = future.result()
                    results[doc_id] = result
                    
                    # Save intermediate results periodically
                    if (i + 1) % 5 == 0:
                        self.save_intermediate_results(results)
                        logger.info(f"Saved intermediate results: {i + 1}/{len(remaining_docs)} completed")
                        
                except Exception as e:
                    logger.error(f"Future failed: {e}")
        
        total_time = time.time() - start_time
        
        # Save final results
        self.save_final_results(results, total_time)
        
        return results
    
    def save_intermediate_results(self, results: Dict):
        """Save intermediate results to file."""
        results_file = self.output_dir / "ner_re_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def save_final_results(self, results: Dict, total_time: float):
        """Save final results with metadata."""
        # Save main results
        results_file = self.output_dir / "ner_re_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Calculate actual processing times
        individual_times = [r.get("processing_time", {}).get("total_time", 0) for r in results.values() if "error" not in r]
        total_individual_time = sum(individual_times)
        
        # Save summary
        summary = {
            "total_documents": len(results),
            "successful_documents": len([r for r in results.values() if "error" not in r]),
            "failed_documents": len([r for r in results.values() if "error" in r]),
            "wall_clock_time": total_time,
            "total_processing_time": total_individual_time,
            "average_time_per_doc": total_individual_time / len([r for r in results.values() if "error" not in r]) if results else 0,
            "parallel_efficiency": (total_individual_time / (total_time * self.max_workers)) * 100 if total_time > 0 else 0,
            "total_entities": sum(len(r.get("entities", [])) for r in results.values() if "error" not in r),
            "total_relations": sum(len(r.get("triples", [])) for r in results.values() if "error" not in r),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_file = self.output_dir / "ner_re_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"NER/RE processing completed:")
        logger.info(f"  Total documents: {summary['total_documents']}")
        logger.info(f"  Successful: {summary['successful_documents']}")
        logger.info(f"  Failed: {summary['failed_documents']}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average time per doc: {summary['average_time_per_doc']:.2f}s")
        logger.info(f"  Total entities: {summary['total_entities']}")
        logger.info(f"  Total relations: {summary['total_relations']}")


class KnowledgeGraphProcessor:
    """Handles knowledge graph construction from NER/RE results."""
    
    def __init__(self, output_dir: str = "output/optimized_pipeline"):
        self.output_dir = Path(output_dir)
        self.settings = Settings()
    
    def build_knowledge_graph(self, ner_re_results: Dict, expected_documents: Optional[int] = None) -> bool:
        """Build knowledge graph from existing NER/RE results."""
        try:
            import igraph as ig
            
            logger.info("🕸️ Building Knowledge Graph from NER/RE results")
            
            # Data consistency check
            if expected_documents and len(ner_re_results) != expected_documents:
                logger.warning(f"⚠️ Data consistency issue detected!")
                logger.warning(f"  Expected documents: {expected_documents}")
                logger.warning(f"  Available NER/RE results: {len(ner_re_results)}")
                logger.warning(f"  Missing documents: {expected_documents - len(ner_re_results)}")
                logger.warning(f"  Knowledge graph will be built from available data only.")
                
                # Check which documents are missing
                available_doc_ids = set(ner_re_results.keys())
                expected_doc_ids = set(f"doc_{i}" for i in range(expected_documents))
                missing_doc_ids = expected_doc_ids - available_doc_ids
                
                if missing_doc_ids:
                    missing_sample = sorted(missing_doc_ids)[:5]
                    logger.warning(f"  Missing doc_ids (sample): {missing_sample}")
                    if len(missing_doc_ids) > 5:
                        logger.warning(f"  ... and {len(missing_doc_ids) - 5} more")
            
            # Initialize graph
            graph = ig.Graph(directed=True)
            vertex_map = {}  # entity_text -> vertex_id
            edge_map = {}    # (subject, predicate, object) -> edge_id
            
            # Statistics
            total_entities = 0
            total_relations = 0
            unique_entities = 0
            unique_relations = 0
            
            start_time = time.time()
            
            # Process each document's NER/RE results
            logger.info(f"Processing {len(ner_re_results)} documents with NER/RE results")
            for doc_id, result in ner_re_results.items():
                if "error" in result:
                    logger.warning(f"Skipping {doc_id} due to error: {result['error']}")
                    continue
                
                entities = result.get("entities", [])
                triples = result.get("triples", [])
                
                total_entities += len(entities)
                total_relations += len(triples)
                
                # Add entities as vertices
                for entity in entities:
                    entity_text = entity.get("text", "")
                    entity_type = entity.get("type", "UNKNOWN")
                    
                    if entity_text and entity_text not in vertex_map:
                        vertex_id = graph.vcount()
                        vertex_map[entity_text] = vertex_id
                        graph.add_vertex(
                            name=entity_text,
                            entity_type=entity_type,
                            first_seen=doc_id,
                            occurrence_count=1,
                            source_docs=[doc_id]
                        )
                        unique_entities += 1
                    elif entity_text in vertex_map:
                        # Update existing vertex
                        vertex_id = vertex_map[entity_text]
                        vertex = graph.vs[vertex_id]
                        vertex["occurrence_count"] += 1
                        if doc_id not in vertex["source_docs"]:
                            vertex["source_docs"].append(doc_id)
                
                # Add relations as edges
                for triple in triples:
                    if len(triple) >= 3:
                        subject, predicate, obj = triple[0], triple[1], triple[2]
                        
                        # Ensure both subject and object exist as vertices
                        for entity_text in [subject, obj]:
                            if entity_text and entity_text not in vertex_map:
                                vertex_id = graph.vcount()
                                vertex_map[entity_text] = vertex_id
                                graph.add_vertex(
                                    name=entity_text,
                                    entity_type="INFERRED",
                                    first_seen=doc_id,
                                    occurrence_count=1,
                                    source_docs=[doc_id]
                                )
                                unique_entities += 1
                        
                        # Add edge
                        if subject in vertex_map and obj in vertex_map:
                            edge_key = (subject, predicate, obj)
                            if edge_key not in edge_map:
                                subject_id = vertex_map[subject]
                                object_id = vertex_map[obj]
                                
                                graph.add_edge(
                                    subject_id, object_id,
                                    predicate=predicate,
                                    first_seen=doc_id,
                                    occurrence_count=1,
                                    source_docs=[doc_id]
                                )
                                edge_map[edge_key] = graph.ecount() - 1
                                unique_relations += 1
                            else:
                                # Update existing edge
                                edge_id = edge_map[edge_key]
                                edge = graph.es[edge_id]
                                edge["occurrence_count"] += 1
                                if doc_id not in edge["source_docs"]:
                                    edge["source_docs"].append(doc_id)
            
            kg_time = time.time() - start_time
            
            # Calculate statistics
            stats = {
                "vertex_count": graph.vcount(),
                "edge_count": graph.ecount(),
                "total_entities_processed": total_entities,
                "total_relations_processed": total_relations,
                "unique_entities": unique_entities,
                "unique_relations": unique_relations,
                "construction_time": kg_time,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add graph-based statistics
            if graph.vcount() > 0:
                degrees = graph.degree()
                stats.update({
                    "max_degree": max(degrees) if degrees else 0,
                    "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
                    "connected_components": len(graph.connected_components()),
                    "density": graph.density() if graph.vcount() > 1 else 0
                })
            
            # Save graph
            graph_file = self.output_dir / "knowledge_graph.graphml"
            try:
                graph.write_graphml(str(graph_file))
                logger.info(f"Knowledge graph saved to: {graph_file}")
                graph_success = True
            except Exception as e:
                logger.error(f"Failed to save graph: {e}")
                graph_success = False
            
            # Save statistics
            stats_file = self.output_dir / "knowledge_graph_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge graph construction completed:")
            logger.info(f"  Construction time: {kg_time:.2f}s")
            logger.info(f"  Vertices: {stats['vertex_count']}")
            logger.info(f"  Edges: {stats['edge_count']}")
            logger.info(f"  Unique entities: {unique_entities}/{total_entities}")
            logger.info(f"  Unique relations: {unique_relations}/{total_relations}")
            logger.info(f"  Statistics saved to: {stats_file}")
            
            return graph_success
                
        except Exception as e:
            logger.error(f"Knowledge graph construction failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class BatchEmbeddingProcessor:
    """Handles batch embedding processing from intermediate files."""
    
    def __init__(self, output_dir: str = "output/optimized_pipeline"):
        self.output_dir = Path(output_dir)
        self.settings = Settings()
        
    def load_ner_re_results(self) -> Tuple[Dict, bool]:
        """Load NER/RE results from intermediate file."""
        results_file = self.output_dir / "ner_re_results.json"
        
        if not results_file.exists():
            logger.error(f"NER/RE results file not found: {results_file}")
            return {}, False
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} NER/RE results")
            return results, True
        except Exception as e:
            logger.error(f"Error loading NER/RE results: {e}")
            return {}, False
    
    def process_embeddings(self, companies: List[Dict], ner_re_results: Dict) -> bool:
        """Process embeddings in batch with duplicate prevention."""
        try:
            # Data consistency check
            if len(ner_re_results) != len(companies):
                logger.warning(f"⚠️ Data consistency issue detected!")
                logger.warning(f"  Companies to embed: {len(companies)}")
                logger.warning(f"  Available NER/RE results: {len(ner_re_results)}")
                logger.warning(f"  Missing NER/RE data: {len(companies) - len(ner_re_results)}")
                logger.warning(f"  Only companies with NER/RE results will be embedded.")
            
            # Prepare documents for VectorIndexer
            vector_documents = []
            doc_ids_seen = set()
            skipped_no_ner = 0
            
            for i, company in enumerate(companies):
                doc_id = f"doc_{i}"
                
                # Skip if this doc_id already processed
                if doc_id in doc_ids_seen:
                    logger.warning(f"Skipping duplicate doc_id: {doc_id}")
                    continue
                
                # Skip if no NER/RE results available
                if doc_id not in ner_re_results:
                    skipped_no_ner += 1
                    if skipped_no_ner <= 3:  # Show first 3 as examples
                        logger.warning(f"Skipping {doc_id} ({company['title']}) - no NER/RE results")
                    continue
                    
                doc_ids_seen.add(doc_id)
                
                doc = {
                    "id": doc_id,
                    "text": company["text"],
                    "title": company["title"],
                    "metadata": {"idx": company["idx"], "source": "optimized_pipeline"}
                }
                vector_documents.append(doc)
            
            if skipped_no_ner > 3:
                logger.warning(f"... and {skipped_no_ner - 3} more documents skipped due to missing NER/RE results")
            
            logger.info(f"Prepared {len(vector_documents)} unique documents for embedding (skipped {skipped_no_ner} without NER/RE)")
            
            # Check for existing table to avoid conflicts
            timestamp = int(time.time())
            table_name = f"optimized_pipeline_{timestamp}"
            
            # Ensure table name is unique
            import lancedb
            temp_db = lancedb.connect(str(self.output_dir / "vector_store"))
            existing_tables = temp_db.table_names()
            counter = 0
            while table_name in existing_tables:
                counter += 1
                table_name = f"optimized_pipeline_{timestamp}_{counter}"
            
            logger.info(f"Using table name: {table_name}")
            
            # Initialize embedding service
            embedding_service = EmbeddingService(
                model_name=self.settings.embedding_model_name,
                batch_size=self.settings.embedding_batch_size
            )
            
            logger.info("Loading embedding model...")
            embedding_service.load_model()
            
            # Initialize vector storage with guaranteed unique table name
            vector_storage = VectorStorage(
                db_path=self.output_dir / "vector_store",
                table_name=table_name
            )
            
            logger.info("Connecting to vector storage...")
            vector_storage.connect()
            
            # Initialize vector indexer
            vector_indexer = VectorIndexer(
                embedding_service=embedding_service,
                vector_storage=vector_storage
            )
            
            # Perform indexing with duplicate check
            logger.info("Starting batch embedding processing...")
            start_time = time.time()
            
            # Create embeddings documents and check for duplicates
            processed_docs = []
            processed_ids = set()
            
            # Use embedding service to process documents with careful ID management
            embedding_docs = vector_indexer._prepare_documents_for_embedding(vector_documents, ner_re_results)
            
            if embedding_docs:
                logger.info(f"Processing {len(embedding_docs)} documents for embedding...")
                
                # Check for potential ID conflicts in embedding_docs preparation
                embedding_doc_ids = set()
                filtered_embedding_docs = []
                
                for doc in embedding_docs:
                    doc_id = doc.get('doc_id', '')
                    chunk_index = doc.get('chunk_index', 0)
                    expected_id = f"{doc_id}_{chunk_index}"
                    
                    if expected_id not in embedding_doc_ids:
                        embedding_doc_ids.add(expected_id)
                        filtered_embedding_docs.append(doc)
                        logger.debug(f"Including document with expected ID: {expected_id}")
                    else:
                        logger.warning(f"Skipping duplicate in embedding_docs preparation: {expected_id}")
                
                logger.info(f"After filtering: {len(filtered_embedding_docs)} unique embedding documents")
                
                processed_docs = embedding_service.process_documents(filtered_embedding_docs)
                
                if processed_docs:
                    # Final duplicate check before storage
                    unique_docs = []
                    seen_ids = set()
                    
                    for doc in processed_docs:
                        doc_id = doc.get('id', '')
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            unique_docs.append(doc)
                        else:
                            logger.warning(f"Removing duplicate processed doc: {doc_id}")
                    
                    removed_count = len(processed_docs) - len(unique_docs)
                    logger.info(f"Storing {len(unique_docs)} unique documents (removed {removed_count} duplicates)")
                    
                    # Store in vector database
                    if not vector_storage.table:
                        vector_storage.create_table(unique_docs[:1])
                        # Only add remaining documents to avoid duplicating the first one
                        if len(unique_docs) > 1:
                            vector_storage.add_documents(unique_docs[1:])
                    else:
                        vector_storage.add_documents(unique_docs)
                    
                    logger.info(f"Successfully stored {len(unique_docs)} documents")
                else:
                    logger.error("Failed to process documents for embedding")
                    return False
            else:
                logger.error("No documents prepared for embedding")
                return False
            
            embedding_time = time.time() - start_time
            
            # Validate results and check for duplicates
            success = False
            vector_count = 0
            
            if hasattr(vector_storage, 'table') and vector_storage.table is not None:
                try:
                    vector_count = len(vector_storage.table)
                    success = vector_count > 0
                    
                    # Check for duplicate IDs in stored data
                    stored_data = vector_storage.table.to_pandas()
                    unique_stored_ids = stored_data['id'].nunique()
                    total_stored = len(stored_data)
                    
                    logger.info(f"Embedding processing completed:")
                    logger.info(f"  Processing time: {embedding_time:.2f}s")
                    logger.info(f"  Vectors stored: {vector_count}")
                    logger.info(f"  Unique IDs: {unique_stored_ids}")
                    logger.info(f"  Duplicates check: {'✅ No duplicates' if unique_stored_ids == total_stored else f'❌ {total_stored - unique_stored_ids} duplicates found'}")
                    logger.info(f"  Success: {'✅' if success else '❌'}")
                    
                    if unique_stored_ids != total_stored:
                        logger.error(f"Duplicate detection failed - found {total_stored - unique_stored_ids} duplicate records")
                        return False
                    
                except Exception as e:
                    logger.error(f"Vector validation failed: {e}")
                    return False
            
            return success
            
        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_optimized_pipeline(corpus_path: str, max_workers: int = 4, max_documents: Optional[int] = None, 
                         resume: bool = True, skip_ner: bool = False, skip_embedding: bool = False, skip_kg: bool = False):
    """Run the optimized two-stage pipeline."""
    logger.info("🚀 Starting Optimized Pipeline")
    logger.info(f"Configuration: max_workers={max_workers}, resume={resume}")
    
    start_time = time.time()
    
    try:
        # Load corpus
        with open(corpus_path) as f:
            all_companies = json.load(f)
        
        # Limit documents if specified
        if max_documents:
            companies = all_companies[:max_documents]
        else:
            companies = all_companies
        
        logger.info(f"Loaded {len(companies)} companies from corpus")
        
        # Stage 1: Parallel NER/RE Processing
        ner_re_results = {}
        if not skip_ner:
            logger.info("🧠 Stage 1: Parallel NER/RE Processing")
            processor = ParallelNERProcessor(max_workers=max_workers)
            ner_re_results = processor.process_documents_parallel(companies, resume=resume)
        else:
            logger.info("⏭️  Skipping NER/RE processing")
            # Load existing results
            embedding_processor = BatchEmbeddingProcessor()
            ner_re_results, loaded = embedding_processor.load_ner_re_results()
            if not loaded:
                logger.error("Cannot skip NER/RE processing - no existing results found")
                return False
        
        # Stage 2: Knowledge Graph Construction
        kg_success = True
        if not skip_kg:
            logger.info("🕸️ Stage 2: Knowledge Graph Construction")
            kg_processor = KnowledgeGraphProcessor()
            kg_success = kg_processor.build_knowledge_graph(ner_re_results, expected_documents=len(companies))
        else:
            logger.info("⏭️  Skipping knowledge graph construction")
        
        # Stage 3: Batch Embedding Processing
        embedding_success = True
        if not skip_embedding:
            logger.info("🔢 Stage 3: Batch Embedding Processing")
            embedding_processor = BatchEmbeddingProcessor()
            embedding_success = embedding_processor.process_embeddings(companies, ner_re_results)
        else:
            logger.info("⏭️  Skipping embedding processing")
        
        total_time = time.time() - start_time
        
        # Final summary
        successful_ner = len([r for r in ner_re_results.values() if "error" not in r])
        failed_ner = len([r for r in ner_re_results.values() if "error" in r])
        
        logger.info(f"\n🎉 Pipeline completed in {total_time:.2f}s")
        logger.info(f"NER/RE: {successful_ner} successful, {failed_ner} failed")
        logger.info(f"Knowledge Graph: {'✅' if kg_success else '❌'}")
        logger.info(f"Embedding: {'✅' if embedding_success else '❌'}")
        
        return successful_ner > 0 and kg_success and embedding_success
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


def main():
    """Main function with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Pipeline with Parallel Processing")
    parser.add_argument("--corpus", default="/home/ryan/workspace/github/AShareRAG/data/corpus.json", 
                       help="Path to corpus file")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to process")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing results")
    parser.add_argument("--skip-ner", action="store_true", help="Skip NER/RE processing")
    parser.add_argument("--skip-kg", action="store_true", help="Skip knowledge graph construction")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding processing")
    
    args = parser.parse_args()
    
    success = run_optimized_pipeline(
        corpus_path=args.corpus,
        max_workers=args.workers,
        max_documents=args.max_docs,
        resume=not args.no_resume,
        skip_ner=args.skip_ner,
        skip_kg=args.skip_kg,
        skip_embedding=args.skip_embedding
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()