#!/usr/bin/env python3
"""
Performance comparison script between original and optimized pipelines.
This script provides side-by-side performance analysis.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_documents(corpus_path: Path, limit: int = 5) -> List[Dict]:
    """Load a smaller set for quick comparison testing."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    documents = []
    for i, doc in enumerate(corpus_data[:limit]):
        formatted_doc = {
            'id': f"test_company_{i:02d}",
            'title': doc.get('title', f'Company {i}'),
            'text': doc.get('text', ''),
            'idx': doc.get('idx', i)
        }
        documents.append(formatted_doc)
    
    return documents


def run_original_pipeline(documents: List[Dict]) -> Dict:
    """Run original pipeline and measure performance."""
    logger.info("🔍 Testing ORIGINAL pipeline...")
    
    from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor
    
    start_time = time.time()
    
    # Original constructor (no optimizations)
    constructor = KnowledgeGraphConstructor()
    
    # Process documents (original method)
    results, graph = constructor.process_documents(documents)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate metrics
    total_entities = sum(len(r.get('entities', [])) for r in results.values())
    total_relations = sum(len(r.get('triples', [])) for r in results.values())
    
    return {
        'processing_time': processing_time,
        'documents_count': len(documents),
        'total_entities': total_entities,
        'total_relations': total_relations,
        'graph_vertices': graph.vcount() if graph else 0,
        'graph_edges': graph.ecount() if graph else 0,
        'avg_time_per_doc': processing_time / len(documents) if documents else 0
    }


def run_optimized_pipeline(documents: List[Dict], max_workers: int = 3) -> Dict:
    """Run optimized pipeline and measure performance."""
    logger.info(f"⚡ Testing OPTIMIZED pipeline (workers: {max_workers})...")
    
    # Import the optimized class
    sys.path.append(str(Path(__file__).parent))
    from test_10_companies_optimized_full import FullOptimizedKnowledgeGraphConstructor
    
    start_time = time.time()
    
    # Optimized constructor (disable embeddings for fair comparison)
    constructor = FullOptimizedKnowledgeGraphConstructor(
        max_workers=max_workers,
        enable_embeddings=False  # Disable for pure NER/RE comparison
    )
    
    # Process documents (optimized method)
    results, graph = constructor.process_documents_full_optimized(documents)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate metrics
    total_entities = sum(len(r.get('entities', [])) for r in results.values())
    total_relations = sum(len(r.get('triples', [])) for r in results.values())
    
    return {
        'processing_time': processing_time,
        'documents_count': len(documents),
        'total_entities': total_entities,
        'total_relations': total_relations,
        'graph_vertices': graph.vcount() if graph else 0,
        'graph_edges': graph.ecount() if graph else 0,
        'avg_time_per_doc': processing_time / len(documents) if documents else 0,
        'max_workers': max_workers
    }


def compare_results(original: Dict, optimized: Dict) -> None:
    """Compare and display performance results."""
    logger.info("\n" + "="*60)
    logger.info("📊 PERFORMANCE COMPARISON RESULTS")
    logger.info("="*60)
    
    # Speedup calculations
    speedup = original['processing_time'] / optimized['processing_time'] if optimized['processing_time'] > 0 else 0
    time_saved = original['processing_time'] - optimized['processing_time']
    
    logger.info("⏱️  TIMING COMPARISON:")
    logger.info(f"   Original Pipeline:     {original['processing_time']:8.2f}s")
    logger.info(f"   Optimized Pipeline:    {optimized['processing_time']:8.2f}s")
    logger.info(f"   Time Saved:            {time_saved:8.2f}s")
    logger.info(f"   Speedup Factor:        {speedup:8.2f}x")
    
    logger.info(f"\n📈 PER-DOCUMENT TIMING:")
    logger.info(f"   Original avg/doc:      {original['avg_time_per_doc']:8.2f}s")
    logger.info(f"   Optimized avg/doc:     {optimized['avg_time_per_doc']:8.2f}s")
    logger.info(f"   Improvement:           {speedup:8.2f}x faster")
    
    logger.info(f"\n📊 DATA PROCESSING:")
    logger.info(f"   Documents processed:   {original['documents_count']}")
    logger.info(f"   Total entities:        {original['total_entities']} vs {optimized['total_entities']}")
    logger.info(f"   Total relations:       {original['total_relations']} vs {optimized['total_relations']}")
    
    logger.info(f"\n🔗 GRAPH CONSTRUCTION:")
    logger.info(f"   Graph vertices:        {original['graph_vertices']} vs {optimized['graph_vertices']}")
    logger.info(f"   Graph edges:           {original['graph_edges']} vs {optimized['graph_edges']}")
    
    logger.info(f"\n⚙️  CONFIGURATION:")
    logger.info(f"   Parallel workers:      {optimized.get('max_workers', 'N/A')}")
    
    # Performance assessment
    logger.info(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if speedup >= 3.0:
        logger.info("   🚀 EXCELLENT: >3x speedup achieved!")
    elif speedup >= 2.0:
        logger.info("   ✅ GOOD: >2x speedup achieved")
    elif speedup >= 1.5:
        logger.info("   👍 MODERATE: >1.5x speedup achieved")
    elif speedup >= 1.1:
        logger.info("   📈 SLIGHT: >1.1x speedup achieved")
    else:
        logger.info("   ⚠️  MINIMAL: Limited speedup achieved")
    
    # Verify data consistency
    entity_diff = abs(original['total_entities'] - optimized['total_entities'])
    relation_diff = abs(original['total_relations'] - optimized['total_relations'])
    
    logger.info(f"\n🔍 DATA CONSISTENCY:")
    if entity_diff == 0 and relation_diff == 0:
        logger.info("   ✅ PERFECT: Identical extraction results")
    elif entity_diff <= 2 and relation_diff <= 2:
        logger.info("   ✅ EXCELLENT: Nearly identical results")
    else:
        logger.info(f"   ⚠️  VARIATION: Entity diff: {entity_diff}, Relation diff: {relation_diff}")


def run_scalability_test(corpus_path: Path) -> None:
    """Test scalability with different document counts."""
    logger.info("\n" + "="*60)
    logger.info("📈 SCALABILITY ANALYSIS")
    logger.info("="*60)
    
    test_sizes = [3, 5, 8]  # Different document counts to test
    
    for size in test_sizes:
        logger.info(f"\n🔍 Testing with {size} documents...")
        
        documents = load_test_documents(corpus_path, limit=size)
        
        # Test original
        try:
            original_results = run_original_pipeline(documents)
            logger.info(f"   Original: {original_results['processing_time']:.2f}s ({original_results['avg_time_per_doc']:.2f}s/doc)")
        except Exception as e:
            logger.error(f"   Original failed: {e}")
            continue
        
        # Test optimized with different worker counts
        for workers in [2, 3, 5]:
            try:
                optimized_results = run_optimized_pipeline(documents, max_workers=workers)
                speedup = original_results['processing_time'] / optimized_results['processing_time']
                logger.info(f"   Optimized ({workers}w): {optimized_results['processing_time']:.2f}s ({optimized_results['avg_time_per_doc']:.2f}s/doc) - {speedup:.2f}x speedup")
            except Exception as e:
                logger.error(f"   Optimized ({workers}w) failed: {e}")


def main():
    """Main comparison execution."""
    try:
        # Setup
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        corpus_path = project_root / "data" / "corpus.json"
        
        logger.info("🎯 KNOWLEDGE GRAPH PIPELINE PERFORMANCE COMPARISON")
        logger.info("="*60)
        logger.info(f"📂 Project: {project_root}")
        logger.info(f"📄 Corpus: {corpus_path}")
        
        if not corpus_path.exists():
            logger.error(f"❌ Corpus file not found: {corpus_path}")
            return 1
        
        # Load test documents (smaller set for quick comparison)
        test_size = 5
        logger.info(f"\n📋 Loading {test_size} documents for comparison...")
        documents = load_test_documents(corpus_path, limit=test_size)
        
        company_names = [doc['title'] for doc in documents]
        logger.info(f"🏢 Test companies: {', '.join(company_names)}")
        
        # Run comparisons
        logger.info(f"\n🚀 Starting performance comparison...")
        
        # Original pipeline
        original_results = run_original_pipeline(documents)
        
        # Optimized pipeline
        optimized_results = run_optimized_pipeline(documents, max_workers=3)
        
        # Compare results
        compare_results(original_results, optimized_results)
        
        # Scalability test
        run_scalability_test(corpus_path)
        
        logger.info(f"\n✅ Performance comparison completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)