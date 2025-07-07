"""Tests for logging configuration and functionality."""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.logging_config import (
    LogConfig,
    PerformanceLogger,
    configure_logging,
    get_logger,
)


class TestLogConfig:
    """Test LogConfig model."""

    def test_default_config(self):
        """Test default log configuration."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.enable_console is True
        assert config.enable_file is True
        assert config.file_path == "logs/app.log"
        assert config.max_bytes == 100 * 1024 * 1024  # 100MB
        assert config.backup_count == 5
        assert config.enable_json is True

    def test_custom_config(self):
        """Test custom log configuration."""
        config = LogConfig(
            level="DEBUG",
            enable_console=False,
            file_path="custom.log",
            max_bytes=1024,
        )
        assert config.level == "DEBUG"
        assert config.enable_console is False
        assert config.file_path == "custom.log"
        assert config.max_bytes == 1024


class TestConfigureLogging:
    """Test logging configuration function."""

    def test_configure_with_defaults(self, tmp_path):
        """Test configure logging with default settings."""
        config = LogConfig(file_path=str(tmp_path / "test.log"))
        configure_logging(config)
        
        # Check root logger configuration
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        
        # Test logging
        logger = logging.getLogger("test_logger")
        logger.info("Test message")
        
        # Check file was created
        log_file = tmp_path / "test.log"
        assert log_file.exists()

    def test_configure_console_only(self):
        """Test configure logging with console only."""
        config = LogConfig(enable_file=False, enable_console=True)
        configure_logging(config)
        
        # Check root logger has console handler
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        assert any(isinstance(h, logging.StreamHandler) 
                  for h in root_logger.handlers)

    def test_configure_file_only(self, tmp_path):
        """Test configure logging with file only."""
        config = LogConfig(
            enable_console=False,
            enable_file=True,
            file_path=str(tmp_path / "file_only.log")
        )
        configure_logging(config)
        
        logger = logging.getLogger()
        # Check that file handler is present
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) 
                  for h in logger.handlers)

    def test_json_logging(self, tmp_path):
        """Test JSON formatted logging."""
        config = LogConfig(
            enable_json=True,
            file_path=str(tmp_path / "json.log"),
            enable_console=False
        )
        configure_logging(config)
        
        logger = logging.getLogger("json_test")
        logger.info("Test JSON", extra={"user": "test", "action": "login"})
        
        # Read and verify JSON log
        log_file = tmp_path / "json.log"
        if log_file.exists():
            with open(log_file) as f:
                line = f.readline()
                if line:
                    data = json.loads(line)
                    assert "message" in data
                    assert data["message"] == "Test JSON"

    def test_log_rotation(self, tmp_path):
        """Test log file rotation."""
        config = LogConfig(
            file_path=str(tmp_path / "rotating.log"),
            max_bytes=1024,  # Small size to trigger rotation
            backup_count=2,
            enable_console=False
        )
        configure_logging(config)
        
        logger = logging.getLogger("rotation_test")
        
        # Write enough data to trigger rotation
        for i in range(100):
            logger.info(f"Test message {i}" * 10)
        
        # Check that backup files exist
        log_files = list(tmp_path.glob("rotating.log*"))
        assert len(log_files) > 1  # Main file plus backups

    def test_create_log_directory(self, tmp_path):
        """Test that log directory is created if it doesn't exist."""
        log_dir = tmp_path / "new_dir"
        config = LogConfig(file_path=str(log_dir / "test.log"))
        
        configure_logging(config)
        
        assert log_dir.exists()
        assert log_dir.is_dir()


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_basic(self):
        """Test getting a basic logger."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_extra(self):
        """Test getting a logger with extra fields."""
        logger = get_logger("test.extra", extra={"component": "test"})
        assert logger.name == "test.extra"


class TestPerformanceLogger:
    """Test PerformanceLogger class."""

    def test_log_timing(self, caplog):
        """Test timing logging."""
        with caplog.at_level(logging.INFO):
            perf_logger = PerformanceLogger("test_op")
            
            # Log component timing
            perf_logger.log_timing("component_a", 100.5)
            perf_logger.log_timing("component_b", 200.3)
            
            # Check logs
            assert "component_a" in caplog.text
            assert "100.5ms" in caplog.text
            assert "component_b" in caplog.text
            assert "200.3ms" in caplog.text

    def test_log_summary(self, caplog):
        """Test performance summary logging."""
        with caplog.at_level(logging.INFO):
            perf_logger = PerformanceLogger("test_summary")
            
            # Add some timings
            perf_logger.log_timing("step1", 50)
            perf_logger.log_timing("step2", 150)
            perf_logger.log_timing("step3", 100)
            
            # Log summary
            perf_logger.log_summary()
            
            # Check summary
            assert "Performance summary" in caplog.text
            assert "total_time" in caplog.text
            assert "300" in caplog.text  # Total time

    def test_context_manager(self, caplog):
        """Test PerformanceLogger as context manager."""
        import time
        
        with caplog.at_level(logging.INFO):
            with PerformanceLogger("test_context") as perf:
                time.sleep(0.01)  # Small delay
                perf.log_timing("inner_op", 5)
            
            # Check that timing was logged
            assert "test_context" in caplog.text
            assert "inner_op" in caplog.text

    def test_get_timings(self):
        """Test getting timings data."""
        perf_logger = PerformanceLogger("test_get")
        
        perf_logger.log_timing("op1", 10)
        perf_logger.log_timing("op2", 20)
        perf_logger.log_timing("op1", 15)  # Second timing for op1
        
        timings = perf_logger.get_timings()
        
        assert "op1" in timings
        assert "op2" in timings
        assert timings["op1"] == [10, 15]
        assert timings["op2"] == [20]
        assert timings["total_time"] == 45

    def test_structured_output(self):
        """Test structured JSON output."""
        perf_logger = PerformanceLogger("test_json", structured=True)
        
        # Mock the logger to capture output
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_timing("test_op", 123.4)
            
            # Check that structured data was logged
            mock_info.assert_called()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)
            
            assert data["operation"] == "test_json"
            assert data["component"] == "test_op"
            assert data["duration_ms"] == 123.4

    def test_error_logging(self, caplog):
        """Test error logging in performance context."""
        with caplog.at_level(logging.ERROR):
            try:
                with PerformanceLogger("test_error") as perf:
                    perf.log_timing("before_error", 10)
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Check that error was logged
            assert "Error in test_error" in caplog.text
            assert "Test error" in caplog.text