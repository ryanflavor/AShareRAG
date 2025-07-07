"""Tests for fact-based Q&A API endpoints."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.intent_detection.router import IntentRouter
from src.pipeline.fact_qa_pipeline import FactQAPipeline


@pytest.fixture
def mock_intent_router():
    """Mock IntentRouter for testing."""
    router = Mock()
    router.route_query = Mock(return_value={
        "intent_type": "fact_qa",
        "confidence": 0.95,
        "metadata": {"company": "腾讯"},
        "route_to": "vector_retriever",
        "detection_method": "keyword",
        "keywords_matched": ["产品", "业务"]
    })
    return router


@pytest.fixture
def mock_fact_qa_pipeline():
    """Mock FactQAPipeline for testing."""
    pipeline = Mock(spec=FactQAPipeline)
    pipeline.process = Mock(return_value={
        "answer": "腾讯的总部位于深圳。",
        "sources": [
            {
                "content": "腾讯公司总部位于深圳市南山区。",
                "company": "腾讯",
                "score": 0.95
            }
        ],
        "processing_time_ms": 1500
    })
    return pipeline


@pytest.fixture
def client(mock_intent_router, mock_fact_qa_pipeline):
    """Create test client with mocked dependencies."""
    import src.server.main
    
    # Set the mocked components directly
    src.server.main.intent_router = mock_intent_router
    src.server.main.fact_qa_pipeline = mock_fact_qa_pipeline
    
    from src.server.main import app
    
    # Create test client with startup/shutdown events
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    src.server.main.intent_router = None
    src.server.main.fact_qa_pipeline = None


class TestFactQAEndpoints:
    """Test suite for fact-based Q&A API endpoints."""

    def test_query_endpoint_success(self, client):
        """Test successful query processing."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "腾讯的总部在哪里？",
                "company": "腾讯",
                "top_k": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "intent" in data
        assert "processing_time_ms" in data
        assert data["intent"] == "fact_qa"
        assert len(data["sources"]) > 0

    def test_query_endpoint_minimal_request(self, client):
        """Test query with minimal required fields."""
        response = client.post(
            "/api/v1/query",
            json={"query": "腾讯的业务有哪些？"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_endpoint_invalid_request(self, client):
        """Test query with invalid request body."""
        response = client.post(
            "/api/v1/query",
            json={}
        )
        
        assert response.status_code == 422  # FastAPI validation error
        data = response.json()
        assert "detail" in data  # FastAPI validation errors structure

    def test_query_endpoint_long_query(self, client):
        """Test query exceeding maximum length."""
        response = client.post(
            "/api/v1/query",
            json={"query": "x" * 501}  # Exceeds 500 char limit
        )
        
        assert response.status_code == 422  # FastAPI validation error
        data = response.json()
        assert "detail" in data

    def test_query_endpoint_special_characters(self, client):
        """Test query with special characters sanitization."""
        response = client.post(
            "/api/v1/query",
            json={"query": "腾讯<script>alert('xss')</script>的产品"}
        )
        
        assert response.status_code == 200
        # Query should be sanitized

    def test_query_endpoint_company_not_found(self, client, mock_intent_router, mock_fact_qa_pipeline):
        """Test query when company is not found."""
        mock_fact_qa_pipeline.process.side_effect = ValueError("Company not found in dataset")
        
        response = client.post(
            "/api/v1/query",
            json={"query": "未知公司的信息"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "COMPANY_NOT_FOUND"

    def test_query_endpoint_pipeline_error(self, client, mock_fact_qa_pipeline):
        """Test query when pipeline encounters error."""
        mock_fact_qa_pipeline.process.side_effect = Exception("Pipeline error")
        
        response = client.post(
            "/api/v1/query",
            json={"query": "腾讯的产品"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "INTERNAL_ERROR"

    def test_query_endpoint_model_unavailable(self, client, mock_fact_qa_pipeline):
        """Test query when model is unavailable."""
        mock_fact_qa_pipeline.process.side_effect = RuntimeError("Model loading failed")
        
        response = client.post(
            "/api/v1/query",
            json={"query": "腾讯的历史"}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "SERVICE_UNAVAILABLE"

    def test_query_endpoint_relationship_discovery_intent(self, client, mock_intent_router):
        """Test query with relationship discovery intent."""
        mock_intent_router.route_query.return_value = {
            "intent_type": "relationship_discovery",
            "confidence": 0.90,
            "metadata": {},
            "route_to": "hybrid_retriever",
            "detection_method": "keyword",
            "keywords_matched": ["相似", "关系"]
        }
        
        response = client.post(
            "/api/v1/query",
            json={"query": "腾讯和阿里巴巴的关系"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert data["detail"]["code"] == "UNSUPPORTED_INTENT"

    def test_query_endpoint_request_size_limit(self, client):
        """Test request body size limit."""
        # Create a request larger than 1MB
        # Using a very long query instead of metadata since metadata is not in the model
        large_query = "x" * (1024 * 1024 + 1)
        
        response = client.post(
            "/api/v1/query",
            json={"query": large_query}
        )
        
        # Should be rejected by either middleware (413) or validation (422)
        assert response.status_code in [413, 422]

    def test_query_endpoint_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request(query):
            return client.post(
                "/api/v1/query",
                json={"query": query}
            )
        
        queries = [f"腾讯的产品{i}" for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, q) for q in queries]
            responses = [f.result() for f in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_api_documentation_available(self, client):
        """Test that API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/api/v1/query" in data["paths"]

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_query_endpoint_invalid_top_k(self, client):
        """Test query with invalid top_k parameter."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "腾讯的业务",
                "top_k": -1
            }
        )
        
        assert response.status_code == 422  # FastAPI validation error
        data = response.json()
        assert "detail" in data

    def test_query_endpoint_non_chinese_query(self, client):
        """Test query in English (should still work)."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What are Tencent's main products?"}
        )
        
        assert response.status_code == 200

    def test_query_endpoint_response_headers(self, client):
        """Test response headers for security."""
        response = client.post(
            "/api/v1/query",
            json={"query": "腾讯的产品"}
        )
        
        # Security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"