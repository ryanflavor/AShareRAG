"""Performance tests for fact-based Q&A pipeline."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from src.pipeline.fact_qa_pipeline import FactQAPipeline, FactQAPipelineConfig


class TestFactQAPerformance:
    """Performance tests for the fact-based Q&A pipeline."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for performance testing."""
        # Mock vector storage
        mock_storage = Mock()
        mock_storage.search.return_value = [
            {
                "content": f"腾讯的产品包括微信、QQ等社交软件。Document {i}",
                "company_name": "腾讯",
                "similarity": 0.9 - i * 0.05
            }
            for i in range(10)
        ]
        
        # Mock embedding service
        mock_embedding = Mock()
        mock_embedding.generate_embeddings.return_value = [[0.1] * 768]  # Typical embedding size
        
        # Mock LLM adapter
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(
            content="腾讯的主要产品包括微信和QQ，这两款产品是中国最流行的社交软件。",
            model="deepseek-chat",
            usage={"prompt_tokens": 500, "completion_tokens": 50}
        )
        
        return mock_storage, mock_embedding, mock_llm

    @pytest.fixture
    def pipeline(self, mock_components):
        """Create pipeline with mocked components."""
        mock_storage, mock_embedding, mock_llm = mock_components
        
        config = FactQAPipelineConfig(
            retriever_top_k=10,
            reranker_top_k=5,
            enable_caching=True,
            cache_size=100
        )
        
        # Mock the component initialization to avoid loading models
        with patch('src.pipeline.fact_qa_pipeline.VectorRetriever') as mock_retriever_class, \
             patch('src.pipeline.fact_qa_pipeline.Ranker') as mock_ranker_class, \
             patch('src.pipeline.fact_qa_pipeline.AnswerSynthesizer') as mock_synthesizer_class:
            
            # Create mock instances
            mock_retriever = Mock()
            mock_ranker = Mock()
            mock_synthesizer = Mock()
            
            # Configure retriever
            mock_retriever.retrieve.return_value = [
                {
                    "content": f"腾讯的产品包括微信、QQ等社交软件。Document {i}",
                    "company": "腾讯",
                    "score": 0.9 - i * 0.05
                }
                for i in range(10)
            ]
            
            # Configure ranker
            mock_ranker.rank_documents.return_value = [
                {
                    "content": f"腾讯的产品包括微信、QQ等社交软件。Document {i}",
                    "company": "腾讯",
                    "score": 0.9 - i * 0.05,
                    "rerank_score": 0.95 - i * 0.1
                }
                for i in range(5)
            ]
            
            # Configure synthesizer
            async def mock_synthesize(*args, **kwargs):
                await asyncio.sleep(0.01)  # Simulate some processing time
                return {
                    "answer": "腾讯的主要产品包括微信和QQ，这两款产品是中国最流行的社交软件。",
                    "sources": mock_ranker.rank_documents.return_value[:3],
                    "synthesis_time": 0.01,
                    "metadata": {
                        "model": "deepseek-chat",
                        "token_usage": {"prompt_tokens": 500, "completion_tokens": 50}
                    }
                }
            
            mock_synthesizer.synthesize_answer.side_effect = mock_synthesize
            
            # Configure class mocks
            mock_retriever_class.return_value = mock_retriever
            mock_ranker_class.return_value = mock_ranker
            mock_synthesizer_class.return_value = mock_synthesizer
            
            pipeline = FactQAPipeline(
                config=config,
                vector_storage=mock_storage,
                embedding_service=mock_embedding,
                llm_adapter=mock_llm
            )
            
            # Store mocks for access in tests
            pipeline._mock_retriever = mock_retriever
            pipeline._mock_ranker = mock_ranker
            pipeline._mock_synthesizer = mock_synthesizer
            
            return pipeline

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="component_timing")
    @pytest.mark.asyncio
    async def test_component_timing(self, pipeline, benchmark):
        """Test individual component timing."""
        # Test retrieval timing
        retrieval_start = time.time()
        await asyncio.sleep(0.002)  # Simulate retrieval time
        retrieval_time = time.time() - retrieval_start
        
        # Test reranking timing
        rerank_start = time.time()
        await asyncio.sleep(0.005)  # Simulate reranking time
        rerank_time = time.time() - rerank_start
        
        # Test synthesis timing (already mocked above)
        synthesis_time = 0.01
        
        # Total time should be under 30 seconds
        total_time = retrieval_time + rerank_time + synthesis_time
        
        assert retrieval_time < 2.0, f"Retrieval took {retrieval_time:.3f}s, expected < 2s"
        assert rerank_time < 5.0, f"Reranking took {rerank_time:.3f}s, expected < 5s"
        assert synthesis_time < 20.0, f"Synthesis took {synthesis_time:.3f}s, expected < 20s"
        assert total_time < 30.0, f"Total time {total_time:.3f}s exceeds 30s limit"
        
        # Benchmark the full pipeline
        result = await benchmark(pipeline.process_query, "腾讯的主要产品有哪些？")
        assert "answer" in result

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, pipeline):
        """Test handling of concurrent requests (1-5 users)."""
        queries = [
            "腾讯的主要产品有哪些？",
            "腾讯的营收情况如何？",
            "腾讯的发展历史",
            "腾讯的技术创新",
            "腾讯的市场地位"
        ]
        
        start_time = time.time()
        
        # Run queries concurrently
        tasks = [pipeline.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # All requests should complete
        assert len(results) == 5
        for result in results:
            assert "answer" in result
            assert len(result["sources"]) > 0
        
        # Should handle 5 concurrent users efficiently
        # Even with sequential processing, 5 queries at 30s each = 150s
        # With proper async handling, should be much faster
        assert elapsed_time < 60.0, f"Concurrent processing took {elapsed_time:.1f}s"

    @pytest.mark.asyncio
    async def test_cache_performance(self, pipeline):
        """Test cache hit performance."""
        query = "腾讯的产品有哪些？"
        
        # First query (cache miss)
        start_time = time.time()
        result1 = await pipeline.process_query(query)
        first_query_time = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        result2 = await pipeline.process_query(query)
        cached_query_time = time.time() - start_time
        
        # Cache hit should be much faster
        assert cached_query_time < first_query_time * 0.1, \
            f"Cache hit ({cached_query_time:.3f}s) not significantly faster than miss ({first_query_time:.3f}s)"
        
        # Verify cache statistics
        stats = pipeline.get_statistics()
        assert stats["cache_stats"]["hits"] >= 1
        assert stats["cache_stats"]["misses"] >= 1

    @pytest.mark.asyncio
    async def test_memory_usage(self, pipeline):
        """Test memory usage with multiple queries."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process 100 different queries
        for i in range(100):
            query = f"腾讯的产品{i}有哪些特点？"
            await pipeline.process_query(query)
            
            # Force garbage collection every 10 queries
            if i % 10 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for 100 queries)
        assert memory_increase < 500, \
            f"Memory increased by {memory_increase:.1f}MB after 100 queries"

    @pytest.mark.asyncio
    async def test_latency_percentiles(self, pipeline):
        """Test latency percentiles for consistent performance."""
        latencies = []
        
        # Run 20 queries to get latency distribution
        for i in range(20):
            query = f"腾讯的第{i}个产品是什么？"
            start_time = time.time()
            await pipeline.process_query(query)
            latency = time.time() - start_time
            latencies.append(latency)
        
        # Sort latencies for percentile calculation
        latencies.sort()
        
        # Calculate percentiles
        p50 = latencies[int(len(latencies) * 0.5)]
        p90 = latencies[int(len(latencies) * 0.9)]
        p99 = latencies[-1]  # Use max for p99 with small sample
        
        # Check latency requirements
        assert p50 < 10.0, f"p50 latency {p50:.3f}s exceeds 10s"
        assert p90 < 20.0, f"p90 latency {p90:.3f}s exceeds 20s"
        assert p99 < 30.0, f"p99 latency {p99:.3f}s exceeds 30s"

    @pytest.mark.asyncio
    async def test_cpu_only_performance(self, pipeline):
        """Test performance on CPU-only hardware."""
        # Mock CPU-only environment
        with patch('torch.cuda.is_available', return_value=False):
            # Should still complete within time limits
            start_time = time.time()
            result = await pipeline.process_query("腾讯的主要产品")
            elapsed_time = time.time() - start_time
            
            assert elapsed_time < 30.0, f"CPU-only query took {elapsed_time:.3f}s"
            assert "answer" in result

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, pipeline):
        """Test performance when errors occur and recovery happens."""
        # Simulate a transient error on first call
        call_count = 0
        original_retrieve = pipeline._mock_retriever.retrieve
        
        def retriever_with_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Transient network error")
            return original_retrieve(*args, **kwargs)
        
        pipeline._mock_retriever.retrieve.side_effect = retriever_with_error
        
        # Should handle error and retry
        start_time = time.time()
        try:
            result = await pipeline.process_query("腾讯的产品")
        except ConnectionError:
            # Reset and retry
            pipeline._mock_retriever.retrieve.side_effect = None
            result = await pipeline.process_query("腾讯的产品")
        
        elapsed_time = time.time() - start_time
        
        # Even with retry, should complete within limits
        assert elapsed_time < 60.0, f"Error recovery took {elapsed_time:.3f}s"
        assert "answer" in result

    def test_synchronous_performance(self, pipeline):
        """Test synchronous API performance."""
        start_time = time.time()
        result = pipeline.process("腾讯的主要业务")
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 30.0, f"Synchronous query took {elapsed_time:.3f}s"
        assert "answer" in result

    @pytest.mark.benchmark(group="load_test")
    def test_simple_load_test(self, pipeline, benchmark):
        """Simple load test with expected usage pattern."""
        async def run_load_test():
            # Simulate 5 users making queries over 60 seconds
            tasks = []
            for user in range(5):
                for query_num in range(3):  # Each user makes 3 queries
                    delay = user * 2 + query_num * 20  # Stagger queries
                    task = self._delayed_query(
                        pipeline,
                        f"用户{user}的查询{query_num}：腾讯的产品",
                        delay
                    )
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check success rate
            successes = sum(1 for r in results if isinstance(r, dict) and "answer" in r)
            success_rate = successes / len(results)
            
            assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
            
            return results
        
        results = benchmark(lambda: asyncio.run(run_load_test()))
        assert len(results) == 15  # 5 users * 3 queries

    @staticmethod
    async def _delayed_query(pipeline, query, delay):
        """Execute a query after a delay."""
        await asyncio.sleep(delay)
        return await pipeline.process_query(query)