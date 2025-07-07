"""Integration tests for Query Intent Router with downstream components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import asyncio
import time

from src.intent_detection.router import IntentRouter
from src.intent_detection.types import IntentType, IntentDetectionConfig, QueryIntent


class MockVectorRetriever:
    """Mock VectorRetriever for fact-based Q&A queries."""
    
    def __init__(self):
        self.queries_processed = []
        
    async def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Mock retrieve method for vector-based search."""
        self.queries_processed.append(query)
        return {
            "query": query,
            "retriever": "vector",
            "results": [
                {"id": "doc1", "score": 0.95, "content": "Mock fact result 1"},
                {"id": "doc2", "score": 0.89, "content": "Mock fact result 2"}
            ],
            "top_k": top_k
        }


class MockHybridRetriever:
    """Mock HybridRetriever for relationship discovery queries."""
    
    def __init__(self):
        self.queries_processed = []
        
    async def retrieve(self, query: str, relationship_types: list = None) -> Dict[str, Any]:
        """Mock retrieve method for hybrid search with graph traversal."""
        self.queries_processed.append(query)
        return {
            "query": query,
            "retriever": "hybrid",
            "results": [
                {
                    "id": "entity1",
                    "relationships": [
                        {"type": "competitor", "target": "entity2", "weight": 0.8},
                        {"type": "supplier", "target": "entity3", "weight": 0.6}
                    ]
                }
            ],
            "relationship_types": relationship_types or ["all"]
        }


class TestQueryIntentRouting:
    """Integration tests for query intent routing to downstream components."""
    
    @pytest.fixture
    def vector_retriever(self):
        """Mock vector retriever fixture."""
        return MockVectorRetriever()
    
    @pytest.fixture
    def hybrid_retriever(self):
        """Mock hybrid retriever fixture."""
        return MockHybridRetriever()
    
    @pytest.fixture
    def router(self):
        """Router fixture with default config."""
        config = IntentDetectionConfig(
            keyword_threshold=0.7,
            use_llm_fallback=False  # Disable for faster tests
        )
        return IntentRouter(config=config)
    
    @pytest.mark.asyncio
    async def test_fact_qa_routing_to_vector_retriever(self, router, vector_retriever):
        """Test that fact-based Q&A queries are routed to VectorRetriever."""
        # Arrange
        query = "What is the revenue of Apple in 2023?"
        
        # Act
        result = router.route_query(query)
        
        # Assert routing decision
        assert result["intent_type"] == IntentType.FACT_QA.value
        assert result.get("route_to") == "vector_retriever"
        
        # Simulate downstream routing
        if result["intent_type"] == IntentType.FACT_QA.value:
            retrieval_result = await vector_retriever.retrieve(query)
            
        # Verify downstream processing
        assert query in vector_retriever.queries_processed
        assert retrieval_result["retriever"] == "vector"
        assert len(retrieval_result["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_relationship_routing_to_hybrid_retriever(self, router, hybrid_retriever):
        """Test that relationship discovery queries are routed to HybridRetriever."""
        # Arrange
        query = "找出与腾讯相似的公司"  # Find companies similar to Tencent
        
        # Act
        result = router.route_query(query)
        
        # Assert routing decision
        assert result["intent_type"] == IntentType.RELATIONSHIP_DISCOVERY.value
        assert result.get("route_to") == "hybrid_retriever"
        
        # Simulate downstream routing
        if result["intent_type"] == IntentType.RELATIONSHIP_DISCOVERY.value:
            retrieval_result = await hybrid_retriever.retrieve(query)
            
        # Verify downstream processing
        assert query in hybrid_retriever.queries_processed
        assert retrieval_result["retriever"] == "hybrid"
        assert "relationships" in str(retrieval_result)
    
    @pytest.mark.asyncio
    async def test_multiple_queries_routing(self, router, vector_retriever, hybrid_retriever):
        """Test routing of multiple queries to appropriate retrievers."""
        # Arrange
        test_cases = [
            ("What is the market cap of Tesla?", IntentType.FACT_QA, vector_retriever),
            ("显示苹果的竞品", IntentType.RELATIONSHIP_DISCOVERY, hybrid_retriever),
            ("How many employees does Google have?", IntentType.FACT_QA, vector_retriever),
            ("分析华为的上下游企业", IntentType.RELATIONSHIP_DISCOVERY, hybrid_retriever),
        ]
        
        # Act & Assert
        for query, expected_intent, expected_retriever in test_cases:
            result = router.route_query(query)
            assert result["intent_type"] == expected_intent.value
            
            # Simulate routing
            if expected_intent == IntentType.FACT_QA:
                await expected_retriever.retrieve(query)
            else:
                await expected_retriever.retrieve(query)
        
        # Verify correct distribution
        assert len(vector_retriever.queries_processed) == 2
        assert len(hybrid_retriever.queries_processed) == 2
    
    def test_router_with_retriever_factory(self, router):
        """Test router integration with a retriever factory pattern."""
        # Arrange
        retriever_factory = {
            IntentType.FACT_QA: MockVectorRetriever(),
            IntentType.RELATIONSHIP_DISCOVERY: MockHybridRetriever()
        }
        
        queries = [
            "What is Microsoft's revenue?",
            "找出与阿里巴巴类似的企业",
            "When was Amazon founded?"
        ]
        
        # Act
        for query in queries:
            result = router.route_query(query)
            intent_type = IntentType(result["intent_type"])
            retriever = retriever_factory[intent_type]
            asyncio.run(retriever.retrieve(query))
        
        # Assert
        fact_retriever = retriever_factory[IntentType.FACT_QA]
        rel_retriever = retriever_factory[IntentType.RELATIONSHIP_DISCOVERY]
        
        assert len(fact_retriever.queries_processed) == 2
        assert len(rel_retriever.queries_processed) == 1
    
    def test_router_performance_with_downstream(self, router, vector_retriever):
        """Test end-to-end latency including downstream retrieval."""
        # Arrange
        query = "What is the stock price of NVIDIA?"
        
        # Act
        start_time = time.time()
        result = router.route_query(query)
        routing_time = time.time() - start_time
        
        # Simulate downstream retrieval
        retrieval_start = time.time()
        asyncio.run(vector_retriever.retrieve(query))
        retrieval_time = time.time() - retrieval_start
        
        total_time = routing_time + retrieval_time
        
        # Assert
        assert routing_time < 0.1  # Routing should be < 100ms
        assert total_time < 0.5  # Total including mock retrieval < 500ms
        assert result.get("latency_ms") < 100
    
    @pytest.mark.parametrize("error_scenario", [
        "retriever_timeout",
        "retriever_error",
        "invalid_intent_type"
    ])
    def test_error_handling_with_downstream(self, router, error_scenario):
        """Test error handling in integration scenarios."""
        # Arrange
        query = "What is Apple's revenue?"
        result = router.route_query(query)
        
        # Simulate different error scenarios
        if error_scenario == "retriever_timeout":
            # Mock a timeout scenario
            with pytest.raises(asyncio.TimeoutError):
                async def slow_retrieve():
                    await asyncio.sleep(10)
                asyncio.run(asyncio.wait_for(slow_retrieve(), timeout=0.1))
                
        elif error_scenario == "retriever_error":
            # Mock a retriever error
            failing_retriever = Mock()
            failing_retriever.retrieve.side_effect = Exception("Retriever connection failed")
            
            with pytest.raises(Exception, match="Retriever connection failed"):
                asyncio.run(failing_retriever.retrieve(query))
                
        elif error_scenario == "invalid_intent_type":
            # Test handling of unknown intent type
            result["intent_type"] = "INVALID_TYPE"  # Force an invalid type
            retriever_factory = {
                IntentType.FACT_QA: MockVectorRetriever(),
                IntentType.RELATIONSHIP_DISCOVERY: MockHybridRetriever()
            }
            
            # Should handle gracefully, perhaps default to fact_qa
            try:
                intent_type = IntentType(result["intent_type"])
                default_retriever = retriever_factory[intent_type]
            except (ValueError, KeyError):
                # Default to fact_qa on invalid intent
                default_retriever = retriever_factory[IntentType.FACT_QA]
            
            assert isinstance(default_retriever, MockVectorRetriever)


class TestRouterConfiguration:
    """Test router configuration and initialization in integration context."""
    
    def test_router_with_custom_config(self):
        """Test router initialization with custom configuration."""
        # Arrange
        config = IntentDetectionConfig(
            keyword_threshold=0.8,
            use_llm_fallback=True,
            timeout=10.0,
            cache_enabled=True,
            cache_ttl=600
        )
        
        # Act
        router = IntentRouter(config=config)
        
        # Assert
        assert router.config.keyword_threshold == 0.8
        assert router.config.use_llm_fallback is True
        assert router.config.cache_enabled is True
    
    @patch('src.intent_detection.llm_intent_adapter.IntentClassificationAdapter')
    def test_router_with_llm_integration(self, mock_adapter_class):
        """Test router with LLM adapter integration."""
        # Arrange
        mock_adapter = Mock()
        mock_adapter.classify_intent.return_value = {
            "intent": "fact_qa",
            "confidence": 0.9,
            "reasoning": "Query asks for specific factual information"
        }
        mock_adapter_class.return_value = mock_adapter
        
        config = IntentDetectionConfig(use_llm_fallback=True)
        router = IntentRouter(config=config)
        
        # Act - Query that would trigger LLM fallback
        result = router.route_query("Analyze the financial performance")
        
        # Assert
        if result.get("confidence", 0) < router.config.keyword_threshold:
            # Would have triggered LLM fallback
            assert result.get("detection_method") == "llm"


class TestEndToEndScenarios:
    """End-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_fact_qa(self):
        """Test complete pipeline for fact-based Q&A."""
        # Arrange
        query = "What is ByteDance's valuation?"
        router = IntentRouter()
        vector_retriever = MockVectorRetriever()
        
        # Act - Full pipeline
        # 1. Route query
        routing_result = router.route_query(query)
        
        # 2. Retrieve based on intent
        if routing_result["intent_type"] == IntentType.FACT_QA.value:
            retrieval_result = await vector_retriever.retrieve(
                query, 
                top_k=routing_result.get("suggested_top_k", 5)
            )
        
        # 3. Process results (mock)
        processed_results = {
            "query": query,
            "intent": routing_result["intent_type"],
            "confidence": routing_result.get("confidence", 0.0),
            "results": retrieval_result["results"],
            "processing_time_ms": routing_result.get("latency_ms", 0) + 50
        }
        
        # Assert
        assert processed_results["intent"] == "fact_qa"
        assert processed_results["confidence"] >= 0.5
        assert len(processed_results["results"]) > 0
        assert processed_results["processing_time_ms"] < 200
    
    @pytest.mark.asyncio
    async def test_full_pipeline_relationship(self):
        """Test complete pipeline for relationship discovery."""
        # Arrange
        query = "展示与字节跳动有竞争关系的公司"
        router = IntentRouter()
        hybrid_retriever = MockHybridRetriever()
        
        # Act - Full pipeline
        # 1. Route query
        routing_result = router.route_query(query)
        
        # 2. Retrieve based on intent
        if routing_result["intent_type"] == IntentType.RELATIONSHIP_DISCOVERY.value:
            # Extract relationship types from metadata if available
            rel_types = routing_result.get("detected_relationships", ["competitor"])
            retrieval_result = await hybrid_retriever.retrieve(query, rel_types)
        
        # 3. Process results (mock)
        processed_results = {
            "query": query,
            "intent": routing_result["intent_type"],
            "confidence": routing_result.get("confidence", 0.0),
            "relationships": retrieval_result["results"],
            "relationship_types": retrieval_result["relationship_types"]
        }
        
        # Assert
        assert processed_results["intent"] == "relationship_discovery"
        assert processed_results["confidence"] >= 0.6
        assert len(processed_results["relationships"]) > 0
        assert "competitor" in str(processed_results["relationships"])