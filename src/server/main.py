"""FastAPI application for AShareRAG fact-based Q&A system."""

import re
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config.settings import settings
from src.intent_detection.router import IntentRouter
from src.pipeline.fact_qa_pipeline import FactQAPipeline
from src.utils.logging_config import (
    LogConfig,
    PerformanceLogger,
    configure_logging,
    get_logger,
)

# Configure logging
log_config = LogConfig(
    level="INFO",
    file_path="logs/api.log",
    enable_json=True,
    max_bytes=100 * 1024 * 1024,  # 100MB
    backup_count=5
)
configure_logging(log_config)
logger = get_logger(__name__)

# Initialize components (lazy loading)
intent_router = None
fact_qa_pipeline = None

# Create FastAPI app
app = FastAPI(
    title="AShareRAG Fact-based Q&A API",
    description="API for querying Chinese A-share company information using RAG",
    version="1.0.0"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="The user's query", min_length=1, max_length=500)
    company: str | None = Field(None, description="Optional company name to filter results")
    top_k: int = Field(10, description="Number of documents to retrieve", ge=1, le=50)

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize query to prevent injection attacks."""
        # Remove HTML tags and script elements
        v = re.sub(r'<[^>]+>', '', v)
        # Remove potentially dangerous characters while preserving Chinese
        v = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef？！。，、；：""'r'（）《》【】+*/=\-]', '', v)
        # Trim whitespace
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty after sanitization")
        return v


class SourceInfo(BaseModel):
    """Information about a source document."""

    content: str = Field(..., description="The source text content")
    company: str = Field(..., description="The company name")
    score: float = Field(..., description="Relevance score", ge=0, le=1)


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str = Field(..., description="The synthesized answer")
    sources: list[SourceInfo] = Field(..., description="Source documents used")
    intent: str = Field(..., description="Detected query intent")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
    details: dict | None = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


# Middleware for request size limiting
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to 1MB."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > 1024 * 1024:  # 1MB limit
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"}
                )
        except ValueError:
            pass

    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response


# API endpoints
@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        404: {"model": ErrorResponse, "description": "Company not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    }
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a fact-based query about Chinese A-share companies.
    
    This endpoint accepts natural language queries and returns structured answers
    with source citations from the knowledge base.
    """
    with PerformanceLogger("query_endpoint", structured=True) as perf:
        try:
            logger.info(f"Processing query: {request.query[:100]}...")

            # Check if components are initialized
            if intent_router is None or fact_qa_pipeline is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Service is not initialized yet",
                        "code": "SERVICE_UNAVAILABLE",
                        "details": {"reason": "Components not initialized"}
                    }
                )

            # Route the query to determine intent
            start_routing = time.time()
            route_result = intent_router.route_query(request.query)
            perf.log_timing("intent_routing", (time.time() - start_routing) * 1000)

            # Check if intent is supported
            if route_result["intent_type"] != "fact_qa":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"Query intent '{route_result['intent_type']}' is not supported by this endpoint",
                        "code": "UNSUPPORTED_INTENT",
                        "details": {"detected_intent": route_result["intent_type"]}
                    }
                )

            # Process fact-based query
            try:
                start_pipeline = time.time()
                result = fact_qa_pipeline.process(
                    query=request.query,
                    company=request.company,
                    top_k=request.top_k
                )
                perf.log_timing("pipeline_processing", (time.time() - start_pipeline) * 1000)
            except ValueError as e:
                if "Company not found" in str(e):
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": str(e),
                            "code": "COMPANY_NOT_FOUND",
                            "details": {"company": request.company}
                        }
                    )
                raise
            except RuntimeError as e:
                if "Model loading failed" in str(e) or "CUDA out of memory" in str(e):
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "error": "Service temporarily unavailable",
                            "code": "SERVICE_UNAVAILABLE",
                            "details": {"reason": str(e)}
                        }
                    )
                raise

            # Get total processing time
            total_time_ms = sum(sum(times) for times in perf.timings.values())

            # Format response
            response = QueryResponse(
                answer=result["answer"],
                sources=[
                    SourceInfo(
                        content=source["content"],
                        company=source["company"],
                        score=source["score"]
                    )
                    for source in result["sources"]
                ],
                intent=route_result["intent_type"],
                processing_time_ms=int(total_time_ms)
            )

            logger.info("Query processed successfully")
            return response

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing query: {e!s}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "An unexpected error occurred",
                    "code": "INTERNAL_ERROR",
                    "details": {"message": str(e)}
                }
            )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the current status of the API service.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AShareRAG Fact-based Q&A API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global intent_router, fact_qa_pipeline

    logger.info("Starting AShareRAG API server...")
    try:
        # Initialize intent router if not already set (for testing)
        if intent_router is None:
            intent_router = IntentRouter()

        # Initialize fact QA pipeline if not already set (for testing)
        if fact_qa_pipeline is None:
            from src.adapters.deepseek_adapter import DeepSeekAdapter
            from src.components.embedding_service import EmbeddingService
            from src.components.vector_storage import VectorStorage
            from src.pipeline.fact_qa_pipeline import FactQAPipelineConfig

            # Create components
            vector_storage = VectorStorage()
            embedding_service = EmbeddingService()
            llm_adapter = DeepSeekAdapter(api_key=settings.deepseek_api_key)

            # Create pipeline config
            pipeline_config = FactQAPipelineConfig(
                retriever_top_k=settings.retriever_top_k,
                reranker_top_k=settings.reranker_top_k,
                relevance_threshold=settings.reranker_relevance_threshold,
                reranker_batch_size=settings.reranker_batch_size,
            )

            # Initialize pipeline
            fact_qa_pipeline = FactQAPipeline(
                config=pipeline_config,
                vector_storage=vector_storage,
                embedding_service=embedding_service,
                llm_adapter=llm_adapter,
            )

        logger.info("API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start API server: {e!s}", exc_info=True)
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")
    try:
        # Cleanup components if needed
        if hasattr(fact_qa_pipeline, 'close'):
            fact_qa_pipeline.close()
        logger.info("API server shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e!s}", exc_info=True)
