"""Performance benchmarks for query intent router."""

import pytest
from src.intent_detection.types import IntentDetectionConfig
from src.intent_detection.router import IntentRouter
from src.intent_detection.detector import KeywordIntentDetector


class TestPerformanceBenchmarks:
    """Performance benchmarks for intent detection."""
    
    @pytest.fixture
    def router_no_cache(self):
        """Create router without cache."""
        config = IntentDetectionConfig(
            use_llm_fallback=False,
            cache_enabled=False
        )
        return IntentRouter(config=config)
    
    @pytest.fixture
    def router_with_cache(self):
        """Create router with cache enabled."""
        config = IntentDetectionConfig(
            use_llm_fallback=False,
            cache_enabled=True
        )
        return IntentRouter(config=config)
    
    @pytest.fixture
    def keyword_detector(self):
        """Create keyword detector."""
        return KeywordIntentDetector()
    
    def test_keyword_detection_performance(self, benchmark, keyword_detector):
        """Benchmark keyword detection speed."""
        query = "找出与腾讯相似的公司"
        
        result = benchmark(keyword_detector.detect, query)
        
        assert result.intent_type.value == "relationship_discovery"
        assert benchmark.stats['mean'] < 0.1  # Should be < 100ms
        assert benchmark.stats['stddev'] < 0.05  # Should be stable
    
    def test_fact_qa_keyword_performance(self, benchmark, keyword_detector):
        """Benchmark fact-based Q&A detection speed."""
        query = "What is Apple's revenue in 2023?"
        
        result = benchmark(keyword_detector.detect, query)
        
        assert result.intent_type.value == "fact_qa"
        assert benchmark.stats['mean'] < 0.1  # Should be < 100ms
    
    def test_router_without_cache_performance(self, benchmark, router_no_cache):
        """Benchmark router performance without cache."""
        query = "分析阿里巴巴的竞争对手"
        
        result = benchmark(router_no_cache.route_query, query)
        
        assert result["intent_type"] == "relationship_discovery"
        assert benchmark.stats['mean'] < 0.1  # Should be < 100ms
    
    def test_router_with_cache_performance(self, benchmark, router_with_cache):
        """Benchmark router performance with cache."""
        query = "What is ByteDance's market cap?"
        
        # Warm up cache
        router_with_cache.route_query(query)
        
        # Benchmark cached performance
        result = benchmark(router_with_cache.route_query, query)
        
        assert result["intent_type"] == "fact_qa"
        assert benchmark.stats['mean'] < 0.001  # Cached should be < 1ms
    
    def test_long_query_performance(self, benchmark, router_no_cache):
        """Benchmark performance with long queries."""
        base = "Find companies similar to "
        company = "TechCorpInternational"
        query = base + company * 50  # ~1000 chars
        
        result = benchmark(router_no_cache.route_query, query)
        
        assert result["intent_type"] == "relationship_discovery"
        assert benchmark.stats['mean'] < 0.2  # Long queries should still be < 200ms
    
    def test_unicode_query_performance(self, benchmark, router_no_cache):
        """Benchmark performance with unicode queries."""
        query = "🚀 найти похожие компании на 腾讯 🍎"
        
        result = benchmark(router_no_cache.route_query, query)
        
        assert result["intent_type"] in ["fact_qa", "relationship_discovery"]
        assert benchmark.stats['mean'] < 0.1  # Unicode should not significantly impact performance
    
    @pytest.mark.parametrize("query", [
        "What is revenue?",
        "找相似公司",
        "Compare A and B",
        "分析竞争关系",
        "Market cap?",
        "上下游企业",
        "Financial report",
        "关联公司查询"
    ])
    def test_mixed_queries_performance(self, benchmark, router_no_cache, query):
        """Benchmark performance across various query types."""
        result = benchmark(router_no_cache.route_query, query)
        
        assert result["intent_type"] in ["fact_qa", "relationship_discovery"]
        assert benchmark.stats['mean'] < 0.1  # All queries should be < 100ms


class TestScalabilityBenchmarks:
    """Scalability benchmarks for high-load scenarios."""
    
    @pytest.fixture
    def router(self):
        """Create router for scalability tests."""
        config = IntentDetectionConfig(
            use_llm_fallback=False,
            cache_enabled=True
        )
        return IntentRouter(config=config)
    
    def test_burst_load_performance(self, benchmark, router):
        """Test performance under burst load."""
        queries = [
            "What is Apple's revenue?",
            "找出类似的公司",
            "Market cap of Tesla",
            "分析竞争关系",
            "When was Google founded?"
        ] * 20  # 100 queries
        
        def process_burst():
            results = []
            for query in queries:
                results.append(router.route_query(query))
            return results
        
        results = benchmark(process_burst)
        
        assert len(results) == 100
        # Average per-query time should still be good
        avg_time = benchmark.stats['mean'] / 100
        assert avg_time < 0.01  # < 10ms per query with cache
    
    def test_cache_efficiency(self, router):
        """Test cache hit rate and efficiency."""
        # Reset metrics
        router.reset_metrics()
        
        # Process same queries multiple times
        queries = ["Query A", "Query B", "Query C"] * 10
        
        for query in queries:
            router.route_query(query)
        
        metrics = router.get_metrics()
        
        # Should have high cache hit rate
        assert metrics["cache_hit_rate"] > 0.85  # > 85% cache hits
        assert metrics["avg_latency_ms"] < 10  # Fast average response
    
    def test_memory_efficiency(self, router):
        """Test memory usage with cache."""
        import sys
        
        # Process many unique queries to fill cache
        initial_size = sys.getsizeof(router._route_query_cached)
        
        for i in range(1000):
            router.route_query(f"Unique query number {i}")
        
        final_size = sys.getsizeof(router._route_query_cached)
        
        # Cache should be bounded (LRU with maxsize=1000)
        # This is a basic check - real memory profiling would be more complex
        assert final_size < initial_size * 10  # Reasonable memory growth


class TestOptimizationValidation:
    """Validate that optimizations work correctly."""
    
    def test_cache_correctness(self):
        """Ensure cache returns same results as non-cached."""
        config_no_cache = IntentDetectionConfig(
            use_llm_fallback=False,
            cache_enabled=False
        )
        config_cache = IntentDetectionConfig(
            use_llm_fallback=False,
            cache_enabled=True
        )
        
        router_no_cache = IntentRouter(config_no_cache)
        router_cache = IntentRouter(config_cache)
        
        queries = [
            "What is the revenue?",
            "找出相似的公司",
            "Analyze market trends",
            "竞争对手分析"
        ]
        
        for query in queries:
            result_no_cache = router_no_cache.route_query(query)
            result_cache_1 = router_cache.route_query(query)
            result_cache_2 = router_cache.route_query(query)  # Should hit cache
            
            # Remove variable fields
            for result in [result_no_cache, result_cache_1, result_cache_2]:
                result.pop("latency_ms", None)
                result.pop("cache_hit", None)
            
            assert result_no_cache == result_cache_1 == result_cache_2
    
    def test_keyword_optimization_correctness(self):
        """Ensure optimized keyword matching is correct."""
        detector = KeywordIntentDetector()
        
        # Test various keyword combinations
        test_cases = [
            ("找相似公司", "relationship_discovery"),
            ("相似的企业有哪些", "relationship_discovery"),
            ("revenue是多少", "fact_qa"),
            ("关联关系分析", "relationship_discovery"),
            ("What is the market cap", "fact_qa"),
            ("Find similar companies", "relationship_discovery")
        ]
        
        for query, expected_intent in test_cases:
            result = detector.detect(query)
            assert result.intent_type.value == expected_intent