#!/usr/bin/env python3
"""
End-to-end test script to demonstrate knowledge graph construction pipeline with 10 companies.
This script validates the complete pipeline from corpus.json to GraphML output.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from src.components.knowledge_graph_constructor import KnowledgeGraphConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_corpus_data(corpus_path: Path, limit: int = 10) -> list[dict]:
    """
    Load and extract limited number of companies from corpus.json.
    
    Args:
        corpus_path: Path to corpus.json file
        limit: Number of companies to extract (default: 10)
        
    Returns:
        List of document dictionaries with id, title, and text
    """
    logger.info(f"Loading corpus data from {corpus_path}")
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        logger.info(f"Loaded {len(corpus_data)} total documents")
        
        # Extract first 'limit' documents and format them
        documents = []
        for i, doc in enumerate(corpus_data[:limit]):
            # Create standardized document format
            formatted_doc = {
                'id': f"company_{i:03d}",  # company_000, company_001, etc.
                'title': doc.get('title', f'Company {i}'),
                'text': doc.get('text', ''),
                'idx': doc.get('idx', i)
            }
            documents.append(formatted_doc)
        
        logger.info(f"Extracted {len(documents)} companies for processing")
        
        # Log company names for verification
        company_names = [doc['title'] for doc in documents]
        logger.info(f"Companies to process: {company_names}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load corpus data: {e}")
        raise


def run_knowledge_graph_pipeline(documents: list[dict]) -> tuple[dict, str]:
    """
    Execute the complete knowledge graph construction pipeline.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Tuple of (processing_results, graph_file_path)
    """
    logger.info("="*60)
    logger.info("STARTING KNOWLEDGE GRAPH CONSTRUCTION PIPELINE")
    logger.info("="*60)
    
    try:
        # Initialize constructor
        logger.info("Initializing Knowledge Graph Constructor...")
        constructor = KnowledgeGraphConstructor()
        
        # Process documents
        logger.info(f"Processing {len(documents)} companies...")
        results, graph = constructor.process_documents(documents)
        
        # Calculate and log statistics
        logger.info("\n" + "="*50)
        logger.info("GRAPH STATISTICS")
        logger.info("="*50)
        
        stats = constructor.calculate_graph_statistics()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Get top entities
        logger.info("\n" + "="*50)
        logger.info("TOP ENTITIES BY DEGREE")
        logger.info("="*50)
        
        top_entities = constructor.get_top_entities_by_degree(k=10)
        for i, entity in enumerate(top_entities, 1):
            logger.info(f"{i:2d}. {entity['name']} (type: {entity['entity_type']}, degree: {entity['degree']})")
        
        # Analyze connected components
        logger.info("\n" + "="*50)
        logger.info("CONNECTED COMPONENTS ANALYSIS")
        logger.info("="*50)
        
        component_analysis = constructor.analyze_connected_components()
        for key, value in component_analysis.items():
            logger.info(f"{key}: {value}")
        
        # Save graph to GraphML
        logger.info("\n" + "="*50)
        logger.info("SAVING GRAPH TO GRAPHML")
        logger.info("="*50)
        
        settings = Settings()
        graph_file = settings.graph_storage_path / "10_companies_test.graphml"
        
        success = constructor.save_graph(graph_file)
        if success:
            logger.info(f"✅ Successfully saved graph to: {graph_file}")
            logger.info(f"📁 File size: {graph_file.stat().st_size / 1024:.2f} KB")
        else:
            logger.error("❌ Failed to save graph")
            return results, ""
        
        # Verify graph can be loaded back
        logger.info("\n" + "="*50)
        logger.info("VERIFYING GRAPH RELOAD")
        logger.info("="*50)
        
        test_constructor = KnowledgeGraphConstructor()
        loaded_graph = test_constructor.load_graph(graph_file)
        
        if loaded_graph:
            logger.info(f"✅ Successfully reloaded graph with {loaded_graph.vcount()} vertices and {loaded_graph.ecount()} edges")
        else:
            logger.error("❌ Failed to reload graph")
        
        return results, str(graph_file)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


def print_processing_summary(results: dict, graph_file: str) -> None:
    """
    Print a comprehensive summary of the processing results.
    
    Args:
        results: Processing results from constructor
        graph_file: Path to saved graph file
    """
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    
    total_entities = 0
    total_triples = 0
    
    for doc_id, doc_results in results.items():
        entities = doc_results.get('entities', [])
        triples = doc_results.get('triples', [])
        total_entities += len(entities)
        total_triples += len(triples)
        
        logger.info(f"📄 {doc_id}: {len(entities)} entities, {len(triples)} relations")
    
    logger.info(f"\n📊 TOTALS:")
    logger.info(f"   • Documents processed: {len(results)}")
    logger.info(f"   • Total entities extracted: {total_entities}")
    logger.info(f"   • Total relations extracted: {total_triples}")
    logger.info(f"   • Graph file: {graph_file}")
    
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("   • Graph saved in GraphML format for visualization")
    logger.info("   • Can be opened in Gephi, Cytoscape, or other graph tools")
    logger.info("   • Metadata file contains additional statistics")
    logger.info("   • Ready for downstream retrieval operations")


def main():
    """Main execution function."""
    try:
        # Setup paths
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        corpus_path = project_root / "data" / "corpus.json"
        
        logger.info(f"🚀 Starting 10-company knowledge graph pipeline test")
        logger.info(f"📂 Project root: {project_root}")
        logger.info(f"📄 Corpus file: {corpus_path}")
        
        # Verify corpus file exists
        if not corpus_path.exists():
            logger.error(f"❌ Corpus file not found: {corpus_path}")
            return 1
        
        # Load documents
        documents = load_corpus_data(corpus_path, limit=10)
        
        # Run pipeline
        results, graph_file = run_knowledge_graph_pipeline(documents)
        
        # Print summary
        print_processing_summary(results, graph_file)
        
        logger.info("\n✅ Pipeline execution completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)