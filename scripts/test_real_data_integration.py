#!/usr/bin/env python3
"""Test script for fact-based QA with real data from optimized pipeline.

This script tests the complete fact-based QA pipeline using the real dataset
with 120 companies from /output/optimized_pipeline.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from src.adapters.deepseek_adapter import DeepSeekAdapter
from src.components.embedding_service import EmbeddingService
from src.components.vector_indexer import VectorIndexer
from src.components.vector_storage import VectorStorage
from src.pipeline.fact_qa_pipeline import FactQAPipeline, FactQAPipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_real_data_fact_qa():
    """Test fact-based QA with real data from 120 companies."""
    
    logger.info("=== Starting Real Data Fact-based QA Integration Test ===")
    
    # 1. Load real dataset info
    pipeline_path = Path("output/optimized_pipeline")
    stats_file = pipeline_path / "ner_re_summary.json"
    
    if not stats_file.exists():
        logger.error(f"Real dataset not found at {stats_file}")
        return
        
    with open(stats_file) as f:
        dataset_stats = json.load(f)
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"  - Total Documents: {dataset_stats['total_documents']}")
    logger.info(f"  - Total Entities: {dataset_stats['total_entities']}")
    logger.info(f"  - Total Relations: {dataset_stats['total_relations']}")
    
    # 2. Setup components with real vector store
    logger.info("Setting up components with real vector store...")
    
    # Use the latest vector store from optimized pipeline
    vector_stores = list((pipeline_path / "vector_store").glob("*.lance"))
    if not vector_stores:
        logger.error("No vector stores found in optimized pipeline")
        return
        
    # Get the latest vector store
    latest_vector_store = max(vector_stores, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using vector store: {latest_vector_store}")
    
    # Initialize components
    embedding_service = EmbeddingService()
    logger.info("Loading embedding model...")
    if not embedding_service.manager.load_model():
        logger.error("Failed to load embedding model")
        return
    
    # Create vector storage pointing to real data
    vector_storage = VectorStorage(
        db_path=latest_vector_store.parent,
        table_name=latest_vector_store.name.replace('.lance', '')
    )
    
    # Initialize DeepSeek adapter
    if not settings.deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY not found in environment")
        logger.info("Please set your DeepSeek API key to test LLM functionality")
        return
    
    llm_adapter = DeepSeekAdapter()
    
    # 3. Create pipeline with real data configuration
    pipeline_config = FactQAPipelineConfig(
        retriever_top_k=10,
        reranker_top_k=5,
        relevance_threshold=0.3,  # Lower threshold for real data
        answer_max_tokens=1000,
        answer_temperature=0.1,   # Lower temperature for factual accuracy
        answer_language="Chinese",
        include_citations=True,
        enable_caching=False      # Disable for testing
    )
    
    pipeline = FactQAPipeline(
        config=pipeline_config,
        vector_storage=vector_storage,
        embedding_service=embedding_service,
        llm_adapter=llm_adapter
    )
    
    # Connect pipeline
    await pipeline.connect()
    
    # Initialize the vector table (opens existing table)
    logger.info("Opening existing vector table...")
    try:
        # Create dummy data to establish schema - this will open existing table if it exists
        vector_storage.create_table([])
        logger.info("Vector table opened successfully")
    except Exception as e:
        logger.error(f"Failed to open vector table: {e}")
        return
    
    # 4. Test queries with real companies from the dataset
    test_queries = [
        "GQY视讯的主要产品有哪些？",
        # "TCL科技的核心业务是什么？",
        # "一汽解放主要从事什么业务？",
        # "ST三圣的医药制造业务包括哪些产品？",
        # "ST世龙的化工产品主要应用在哪些领域？",
        # "ST东园的新能源业务是怎样的？",
        # "七匹狼的主营业务是什么？",
        # "一心堂主要经营哪类产品？"
    ]
    
    logger.info(f"Testing {len(test_queries)} queries with real data...")
    
    results = []
    total_start_time = time.time()
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n--- Query {i}/{len(test_queries)}: {query} ---")
        
        try:
            start_time = time.time()
            result = await pipeline.process_query(query)
            end_time = time.time()
            
            logger.info(f"Query completed in {end_time - start_time:.2f}s")
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Sources found: {len(result['sources'])}")
            logger.info(f"Retrieval count: {result['metadata']['retrieval_count']}")
            logger.info(f"Reranked count: {result['metadata']['reranked_count']}")
            
            # Store result for summary
            results.append({
                "query": query,
                "answer": result["answer"],
                "sources_count": len(result["sources"]),
                "retrieval_count": result["metadata"]["retrieval_count"],
                "reranked_count": result["metadata"]["reranked_count"],
                "processing_time": end_time - start_time,
                "total_pipeline_time": result["metadata"]["total_time"]
            })
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            results.append({
                "query": query,
                "error": str(e),
                "processing_time": 0
            })
    
    total_end_time = time.time()
    
    # 5. Generate summary report
    logger.info("\n=== REAL DATA INTEGRATION TEST SUMMARY ===")
    logger.info(f"Total test time: {total_end_time - total_start_time:.2f}s")
    logger.info(f"Queries processed: {len([r for r in results if 'error' not in r])}")
    logger.info(f"Queries failed: {len([r for r in results if 'error' in r])}")
    
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_retrieval = sum(r['retrieval_count'] for r in successful_results) / len(successful_results)
        avg_reranked = sum(r['reranked_count'] for r in successful_results) / len(successful_results)
        
        logger.info(f"Average processing time: {avg_time:.2f}s")
        logger.info(f"Average retrieval count: {avg_retrieval:.1f}")
        logger.info(f"Average reranked count: {avg_reranked:.1f}")
    
    # 6. Show detailed results
    logger.info("\n=== DETAILED RESULTS ===")
    for i, result in enumerate(results, 1):
        logger.info(f"\nQuery {i}: {result['query']}")
        if 'error' in result:
            logger.info(f"  ERROR: {result['error']}")
        else:
            logger.info(f"  Answer: {result['answer'][:150]}...")
            logger.info(f"  Sources: {result['sources_count']}")
            logger.info(f"  Time: {result['processing_time']:.2f}s")
    
    # 7. Get pipeline statistics
    stats = pipeline.get_statistics()
    logger.info(f"\n=== PIPELINE STATISTICS ===")
    logger.info(f"Total queries processed: {stats['total_queries']}")
    logger.info(f"Component stats: {json.dumps(stats['component_stats'], indent=2)}")
    
    # Cleanup - vector storage doesn't have disconnect method
    logger.info("Cleaning up pipeline...")
    
    logger.info("\n=== Real Data Integration Test Complete ===")


def main():
    """Main function to run the test."""
    try:
        asyncio.run(test_real_data_fact_qa())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()