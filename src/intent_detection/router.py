"""Intent router implementation."""

import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from .detector import KeywordIntentDetector, LLMIntentDetector
from .types import IntentDetectionConfig, IntentType, QueryIntent

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Structured logger for classification decisions."""

    def __init__(self, logger_name: str = "intent_router"):
        """Initialize structured logger.

        Args:
            logger_name: Base name for the logger.
        """
        self.logger = logging.getLogger(f"{logger_name}.structured")

    def log_classification(self, log_data: dict[str, Any]):
        """Log classification decision in structured format."""
        log_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.logger.info(json.dumps(log_data))

    def log_performance_metrics(self, metrics: dict[str, Any]):
        """Log performance metrics in structured format.

        Args:
            metrics: Dictionary of performance metrics to log.
        """
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        metrics["type"] = "performance_metrics"
        self.logger.info(json.dumps(metrics))

    def log_error(self, error_data: dict[str, Any]):
        """Log errors in structured format."""
        error_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        error_data["type"] = "error"
        self.logger.error(json.dumps(error_data))


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self._metrics = {
            "total_queries": 0,
            "keyword_detections": 0,
            "llm_detections": 0,
            "total_latency_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "latencies": [],
        }

    def record_query(
        self, latency_ms: float, detection_method: str, cache_hit: bool = False
    ):
        """Record query metrics."""
        self._metrics["total_queries"] += 1
        self._metrics["total_latency_ms"] += latency_ms
        self._metrics["latencies"].append(latency_ms)

        if detection_method == "keyword":
            self._metrics["keyword_detections"] += 1
        elif detection_method == "llm":
            self._metrics["llm_detections"] += 1

        if cache_hit:
            self._metrics["cache_hits"] += 1
        else:
            self._metrics["cache_misses"] += 1

    def record_error(self):
        """Record an error occurrence."""
        self._metrics["errors"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        metrics = self._metrics.copy()

        # Calculate averages
        if metrics["total_queries"] > 0:
            metrics["avg_latency_ms"] = (
                metrics["total_latency_ms"] / metrics["total_queries"]
            )
            metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["total_queries"]
        else:
            metrics["avg_latency_ms"] = 0.0
            metrics["cache_hit_rate"] = 0.0

        # Calculate percentiles if we have latencies
        if self._metrics["latencies"]:
            sorted_latencies = sorted(self._metrics["latencies"])
            n = len(sorted_latencies)
            metrics["p50_latency_ms"] = sorted_latencies[n // 2]
            metrics["p95_latency_ms"] = sorted_latencies[int(n * 0.95)]
            metrics["p99_latency_ms"] = sorted_latencies[int(n * 0.99)]

        # Remove raw latencies from output
        metrics.pop("latencies", None)

        return metrics

    def reset_metrics(self):
        """Reset all metrics."""
        self.__init__()


class IntentRouter:
    """Routes queries based on detected intent."""

    def __init__(self, config: IntentDetectionConfig | None = None):
        """Initialize intent router.

        Args:
            config: Optional configuration for intent detection.
        """
        self.config = config or IntentDetectionConfig()
        self._keyword_detector = KeywordIntentDetector(config)
        self._llm_detector = None
        self._performance_monitor = PerformanceMonitor()
        self._structured_logger = StructuredLogger()
        self._cache_enabled = self.config.cache_enabled

        # Initialize cache if enabled
        if self.config.cache_enabled:
            self._route_query_cached = lru_cache(maxsize=1000)(self._route_query_impl)
        else:
            self._route_query_cached = self._route_query_impl

        logger.info(
            f"IntentRouter initialized with config: keyword_threshold={self.config.keyword_threshold}, "
            f"use_llm_fallback={self.config.use_llm_fallback}, cache_enabled={self.config.cache_enabled}"
        )

    def route_query(self, query: str) -> dict[str, Any]:
        """Route query based on detected intent."""
        start_time = time.time()
        cache_hit = False

        try:
            # Check if result is from cache
            if self._cache_enabled and hasattr(self._route_query_cached, "cache_info"):
                cache_info_before = self._route_query_cached.cache_info()

            result = self._route_query_cached(query)

            # Determine if it was a cache hit
            if self._cache_enabled and hasattr(self._route_query_cached, "cache_info"):
                cache_info_after = self._route_query_cached.cache_info()
                cache_hit = cache_info_after.hits > cache_info_before.hits

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            result["latency_ms"] = latency_ms
            result["cache_hit"] = cache_hit

            # Record metrics
            self._performance_monitor.record_query(
                latency_ms=latency_ms,
                detection_method=result["detection_method"],
                cache_hit=cache_hit,
            )

            # Log the routing decision
            logger.info(
                f"Query routed: intent={result['intent_type']}, "
                f"confidence={result['confidence']:.2f}, "
                f"method={result['detection_method']}, "
                f"latency={latency_ms:.1f}ms, "
                f"cache_hit={cache_hit}"
            )

            # Structured logging
            self._structured_logger.log_classification(
                {
                    "query": (query[:100] if query else ""),  # Truncate long queries
                    "intent_type": result["intent_type"],
                    "confidence": result["confidence"],
                    "detection_method": result["detection_method"],
                    "keywords_matched": result.get("keywords_matched", []),
                    "route_to": result["route_to"],
                    "latency_ms": latency_ms,
                    "cache_hit": cache_hit,
                    "hint": result.get("hint"),
                }
            )

            return result

        except Exception as e:
            self._performance_monitor.record_error()

            # Log error details
            logger.error(f"Error routing query: {e}", exc_info=True)

            # Structured error logging
            self._structured_logger.log_error(
                {
                    "query": (query[:100] if query else ""),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "latency_ms": (time.time() - start_time) * 1000,
                }
            )

            raise

    def _route_query_impl(self, query: str) -> dict[str, Any]:
        """Implement query routing logic."""
        # Input validation and sanitization
        query = self._validate_and_sanitize_query(query)

        logger.debug(f"Processing query: {query[:50]}...")

        # Handle empty query after sanitization
        if not query:
            logger.warning("Empty query received")
            keyword_result = self._keyword_detector.detect("")
            result = self._format_routing_result(keyword_result)
            result["hint"] = "Empty query provided. Defaulting to fact-based Q&A."
            return result

        # First try keyword detection
        keyword_start = time.time()
        keyword_result = self._keyword_detector.detect(query)
        keyword_time = (time.time() - keyword_start) * 1000

        logger.debug(
            f"Keyword detection completed in {keyword_time:.1f}ms, "
            f"confidence={keyword_result.confidence:.2f}"
        )

        # Check if confidence meets threshold
        if keyword_result.confidence >= self.config.keyword_threshold:
            logger.info(f"Using keyword detection result for query: {query[:50]}...")
            return self._format_routing_result(keyword_result)

        # Try LLM fallback if enabled and keyword confidence is low
        if self.config.use_llm_fallback:
            if not self._llm_detector:
                # Lazy initialization of LLM detector
                self._init_llm_detector()

            try:
                logger.info(
                    f"Falling back to LLM classification for query: {query[:50]}..."
                )
                llm_start = time.time()
                llm_result = self._llm_detector.detect(query)
                llm_time = (time.time() - llm_start) * 1000

                logger.debug(
                    f"LLM detection completed in {llm_time:.1f}ms, "
                    f"confidence={llm_result.confidence:.2f}"
                )

                if llm_result.confidence >= self.config.llm_threshold:
                    return self._format_routing_result(llm_result)
                else:
                    logger.info(
                        f"LLM confidence {llm_result.confidence:.2f} below threshold "
                        f"{self.config.llm_threshold}, using keyword result"
                    )

            except TimeoutError:
                logger.warning(
                    f"LLM timeout for query: {query[:50]}... after {self.config.timeout}s"
                )
            except Exception as e:
                logger.error(
                    f"LLM error for query '{query[:50]}...': {type(e).__name__}: {e!s}"
                )

        # Default to keyword result even with low confidence
        logger.info(f"Using low-confidence keyword result for query: {query[:50]}...")
        result = self._format_routing_result(keyword_result)
        result["hint"] = "Low confidence detection. Consider rephrasing the query."
        return result

    def _format_routing_result(self, intent: QueryIntent) -> dict[str, Any]:
        """Format the routing result."""
        # Determine target component based on intent type
        if intent.intent_type == IntentType.RELATIONSHIP_DISCOVERY:
            route_to = "hybrid_retriever"
        elif intent.intent_type == IntentType.FACT_QA:
            route_to = "vector_retriever"
        else:
            route_to = "vector_retriever"  # Default

        return {
            "query": intent.query,
            "intent_type": intent.intent_type.value,
            "confidence": intent.confidence,
            "detection_method": intent.detection_method,
            "keywords_matched": intent.keywords_matched,
            "route_to": route_to,
            "metadata": intent.metadata or {},
        }

    def _init_llm_detector(self):
        """Initialize LLM detector lazily."""
        from .llm_intent_adapter import IntentClassificationAdapter

        logger.info("Initializing LLM detector with DeepSeek adapter")
        adapter = IntentClassificationAdapter(enable_cache=self.config.cache_enabled)
        self._llm_detector = LLMIntentDetector(llm_adapter=adapter, config=self.config)

    def get_metrics(self) -> dict[str, Any]:
        """Get routing metrics."""
        return self._performance_monitor.get_metrics()

    def reset_metrics(self):
        """Reset performance metrics."""
        self._performance_monitor.reset_metrics()
        logger.info("Performance metrics reset")

    def log_metrics_summary(self):
        """Log current metrics summary."""
        metrics = self.get_metrics()
        self._structured_logger.log_performance_metrics(metrics)

        # Also log human-readable summary
        logger.info(
            f"Performance Summary: "
            f"Total queries: {metrics['total_queries']}, "
            f"Avg latency: {metrics['avg_latency_ms']:.1f}ms, "
            f"Cache hit rate: {metrics['cache_hit_rate']:.1%}, "
            f"Errors: {metrics['errors']}"
        )

    def _validate_and_sanitize_query(self, query: str) -> str:
        """Validate and sanitize input query."""
        # Handle None
        if query is None:
            return ""

        # Ensure string type
        if not isinstance(query, str):
            query = str(query)

        # Remove leading/trailing whitespace
        query = query.strip()

        # Handle very long queries
        max_length = 10000
        if len(query) > max_length:
            logger.warning(
                f"Query too long ({len(query)} chars), truncating to {max_length}"
            )
            query = query[:max_length]

        # Remove zero-width characters that can cause issues
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"]
        for char in zero_width_chars:
            query = query.replace(char, "")

        # Normalize excessive whitespace
        query = " ".join(query.split())

        # Log potential security concerns (but still process them)
        suspicious_patterns = [
            "<script",
            "</script>",
            "DROP TABLE",
            "DELETE FROM",
            "../",
            "..\\",
            "${",
            "#{",
            "eval(",
            "exec(",
        ]

        for pattern in suspicious_patterns:
            if pattern.lower() in query.lower():
                logger.warning(f"Potentially suspicious pattern detected: {pattern}")
                break

        return query
