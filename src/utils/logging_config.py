"""Logging configuration and utilities for AShareRAG."""

import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LogConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for logs"
    )
    enable_console: bool = Field(default=True, description="Enable console output")
    enable_file: bool = Field(default=True, description="Enable file output")
    file_path: str = Field(default="logs/app.log", description="Log file path")
    max_bytes: int = Field(default=100 * 1024 * 1024, description="Max file size (100MB)")
    backup_count: int = Field(default=5, description="Number of backup files")
    enable_json: bool = Field(default=True, description="Enable JSON formatting")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            for key, value in record.extra.items():
                if key not in log_data:
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def configure_logging(config: LogConfig | None = None) -> None:
    """Configure logging for the application.
    
    Args:
        config: Logging configuration. Uses defaults if not provided.
    """
    if config is None:
        config = LogConfig()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if config.enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(config.format, config.date_format)
            )
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if config.enable_file:
        # Create log directory if it doesn't exist
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8"
        )

        if config.enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(config.format, config.date_format)
            )
        root_logger.addHandler(file_handler)

    # Set level for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str, extra: dict[str, Any] | None = None) -> logging.Logger:
    """Get a logger instance with optional extra fields.
    
    Args:
        name: Logger name (usually __name__)
        extra: Optional extra fields to include in all logs
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if extra:
        # Create adapter to include extra fields
        logger = logging.LoggerAdapter(logger, extra)

    return logger


class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, operation: str, structured: bool = False):
        """Initialize performance logger.
        
        Args:
            operation: Name of the operation being timed
            structured: Whether to output structured JSON logs
        """
        self.operation = operation
        self.structured = structured
        self.logger = get_logger(f"performance.{operation}")
        self.timings: dict[str, list[float]] = {}
        self.start_time = time.time()

    def log_timing(self, component: str, duration_ms: float) -> None:
        """Log timing for a component.
        
        Args:
            component: Component name
            duration_ms: Duration in milliseconds
        """
        if component not in self.timings:
            self.timings[component] = []
        self.timings[component].append(duration_ms)

        if self.structured:
            log_data = {
                "operation": self.operation,
                "component": component,
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.info(
                f"[{self.operation}] {component}: {duration_ms:.1f}ms"
            )

    def get_timings(self) -> dict[str, Any]:
        """Get all timings data.
        
        Returns:
            Dictionary with timing data
        """
        total_time = sum(
            sum(times) for times in self.timings.values()
        )
        return {
            **self.timings,
            "total_time": total_time
        }

    def log_summary(self) -> None:
        """Log performance summary."""
        timings = self.get_timings()
        total_time = timings.pop("total_time", 0)

        if self.structured:
            summary = {
                "operation": self.operation,
                "type": "summary",
                "total_time_ms": total_time,
                "components": {
                    name: {
                        "count": len(times),
                        "total_ms": sum(times),
                        "avg_ms": sum(times) / len(times) if times else 0
                    }
                    for name, times in timings.items()
                }
            }
            self.logger.info(json.dumps(summary))
        else:
            self.logger.info(
                f"Performance summary for {self.operation}: "
                f"total_time={total_time:.1f}ms, "
                f"components={list(timings.keys())}"
            )

    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        elapsed_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.error(
                f"Error in {self.operation}: {exc_type.__name__}: {exc_val}"
            )
        else:
            self.log_timing("total", elapsed_ms)
            self.log_summary()


# Initialize default logging on import
configure_logging()
