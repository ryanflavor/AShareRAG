"""Monitoring and structured logging for intent detection."""

import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from .types import IntentType, QueryIntent


class StructuredLogger:
    """Provides structured JSON logging for intent detection."""

    def __init__(self, logger_name: str = "intent_detection"):
        self.logger = logging.getLogger(logger_name)

    def log_intent_detection(
        self,
        query: str,
        result: QueryIntent,
        latency_ms: float,
        cache_hit: bool = False,
        error: Exception | None = None,
    ):
        """Log intent detection with structured format."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "intent_detection",
            "query": query[:200],  # Truncate long queries
            "query_length": len(query),
            "intent_type": result.intent_type.value if result else None,
            "confidence": result.confidence if result else None,
            "detection_method": result.detection_method if result else None,
            "keywords_matched": result.keywords_matched if result else [],
            "latency_ms": round(latency_ms, 2),
            "cache_hit": cache_hit,
            "success": error is None,
            "error": str(error) if error else None,
        }

        if error:
            self.logger.error(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))

    def log_routing_decision(
        self,
        query: str,
        intent_type: str,
        route_to: str,
        confidence: float,
        hint: str | None = None,
    ):
        """Log routing decision with structured format."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "routing_decision",
            "query": query[:200],
            "intent_type": intent_type,
            "route_to": route_to,
            "confidence": round(confidence, 3),
            "hint": hint,
        }

        self.logger.info(json.dumps(log_entry))

    def log_performance_metrics(self, metrics: dict[str, Any]):
        """Log performance metrics."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "performance_metrics",
            **metrics,
        }

        self.logger.info(json.dumps(log_entry))


class PerformanceMonitor:
    """Monitors performance metrics for intent detection."""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "keyword_detections": 0,
            "llm_detections": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "intent_distribution": {
                IntentType.FACT_QA.value: 0,
                IntentType.RELATIONSHIP_DISCOVERY.value: 0,
                IntentType.UNKNOWN.value: 0,
            },
            "confidence_histogram": {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0,
            },
        }

    def record_detection(
        self,
        intent: QueryIntent,
        latency_ms: float,
        cache_hit: bool = False,
        error: Exception | None = None,
    ):
        """Record a detection event."""
        self.metrics["total_queries"] += 1
        self.metrics["total_latency_ms"] += latency_ms

        if error:
            self.metrics["errors"] += 1
            return

        if cache_hit:
            self.metrics["cache_hits"] += 1

        if intent.detection_method == "keyword":
            self.metrics["keyword_detections"] += 1
        elif intent.detection_method == "llm":
            self.metrics["llm_detections"] += 1

        # Update intent distribution
        self.metrics["intent_distribution"][intent.intent_type.value] += 1

        # Update confidence histogram
        confidence_bucket = self._get_confidence_bucket(intent.confidence)
        self.metrics["confidence_histogram"][confidence_bucket] += 1

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence histogram bucket."""
        if confidence < 0.2:
            return "0.0-0.2"
        elif confidence < 0.4:
            return "0.2-0.4"
        elif confidence < 0.6:
            return "0.4-0.6"
        elif confidence < 0.8:
            return "0.6-0.8"
        else:
            return "0.8-1.0"

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        metrics = self.metrics.copy()

        # Calculate derived metrics
        if metrics["total_queries"] > 0:
            metrics["average_latency_ms"] = round(
                metrics["total_latency_ms"] / metrics["total_queries"], 2
            )
            metrics["cache_hit_rate"] = round(
                metrics["cache_hits"] / metrics["total_queries"], 3
            )
            metrics["error_rate"] = round(
                metrics["errors"] / metrics["total_queries"], 3
            )
            metrics["keyword_detection_rate"] = round(
                metrics["keyword_detections"] / metrics["total_queries"], 3
            )
            metrics["llm_detection_rate"] = round(
                metrics["llm_detections"] / metrics["total_queries"], 3
            )

        return metrics

    def reset_metrics(self):
        """Reset all metrics."""
        self.__init__()


def monitor_performance(method: Callable) -> Callable:
    """Decorator to monitor method performance."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        error = None
        result = None

        try:
            result = method(self, *args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

            # Log performance if instance has logger
            if hasattr(self, "_structured_logger"):
                method_name = method.__name__
                log_entry = {
                    "method": method_name,
                    "latency_ms": round(latency_ms, 2),
                    "success": error is None,
                    "error": str(error) if error else None,
                }

                if error:
                    self._structured_logger.logger.error(
                        f"Method {method_name} failed: {json.dumps(log_entry)}"
                    )
                elif latency_ms > 1000:  # Log slow operations
                    self._structured_logger.logger.warning(
                        f"Slow operation: {json.dumps(log_entry)}"
                    )

    return wrapper
