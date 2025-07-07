#!/usr/bin/env python3
"""Quick script to show the actual document sources."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from src.adapters.deepseek_adapter import DeepSeekAdapter
from src.components.embedding_service import EmbeddingService
from src.components.vector_storage import VectorStorage
from src.pipeline.fact_qa_pipeline import FactQAPipeline, FactQAPipelineConfig


async def show_sources():
    """Show the actual document sources for a query."""
    
    # Setup components (simplified)
    embedding_service = EmbeddingService()
    embedding_service.manager.load_model()
    
    vector_storage = VectorStorage(
        db_path=Path("output/optimized_pipeline/vector_store"),
        table_name="optimized_pipeline_1751890437"
    )
    
    llm_adapter = DeepSeekAdapter()
    
    # Create pipeline
    pipeline_config = FactQAPipelineConfig(
        retriever_top_k=10,
        reranker_top_k=5,
        relevance_threshold=0.3,
        answer_max_tokens=1000,
        answer_temperature=0.1,
        answer_language="Chinese",
        include_citations=True,
        enable_caching=False
    )
    
    pipeline = FactQAPipeline(
        config=pipeline_config,
        vector_storage=vector_storage,
        embedding_service=embedding_service,
        llm_adapter=llm_adapter
    )
    
    await pipeline.connect()
    vector_storage.create_table([])
    
    # Test query
    query = "GQY视讯的主要产品有哪些？"
    result = await pipeline.process_query(query)
    
    print(f"Query: {query}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"\n=== DOCUMENT SOURCES ({len(result['sources'])}) ===")
    
    for i, source in enumerate(result['sources'], 1):
        print(f"\n--- Document {i} ---")
        print(f"Content (first 300 chars): {source.get('content', '')[:300]}...")
        print(f"Company: {source.get('company', 'N/A')}")
        print(f"Score: {source.get('score', 'N/A')}")
        if 'metadata' in source:
            print(f"Metadata: {source['metadata']}")


if __name__ == "__main__":
    asyncio.run(show_sources())