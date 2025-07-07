"""Simple console progress output for user feedback."""

import sys
import time


class ProgressLogger:
    """Simple progress logger for console output."""

    def __init__(self, operation: str):
        """Initialize progress logger.
        
        Args:
            operation: Name of the operation being performed
        """
        self.operation = operation
        self.start_time = time.time()
        self.current_step = None

    def start(self, message: str) -> None:
        """Start a new operation with a message.
        
        Args:
            message: Initial message to display
        """
        self.start_time = time.time()
        print(f"\n🔍 {self.operation}: {message}")
        sys.stdout.flush()

    def update(self, step: str, message: str | None = None) -> None:
        """Update progress with current step.
        
        Args:
            step: Current step name
            message: Optional additional message
        """
        self.current_step = step
        elapsed = time.time() - self.start_time

        if message:
            print(f"  ⏳ [{elapsed:.1f}s] {step}: {message}")
        else:
            print(f"  ⏳ [{elapsed:.1f}s] {step}...")
        sys.stdout.flush()

    def complete(self, message: str | None = None) -> None:
        """Mark operation as complete.
        
        Args:
            message: Optional completion message
        """
        elapsed = time.time() - self.start_time

        if message:
            print(f"  ✅ [{elapsed:.1f}s] Complete: {message}")
        else:
            print(f"  ✅ [{elapsed:.1f}s] {self.operation} completed successfully")
        sys.stdout.flush()

    def error(self, error_message: str) -> None:
        """Log an error.
        
        Args:
            error_message: Error message to display
        """
        elapsed = time.time() - self.start_time
        print(f"  ❌ [{elapsed:.1f}s] Error: {error_message}")
        sys.stdout.flush()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is not None:
            self.error(f"{exc_type.__name__}: {exc_val}")
        else:
            if self.current_step:
                self.complete()


def create_progress_logger(operation: str, enabled: bool = True) -> ProgressLogger | None:
    """Create a progress logger if enabled.
    
    Args:
        operation: Operation name
        enabled: Whether to create logger
        
    Returns:
        ProgressLogger instance or None
    """
    return ProgressLogger(operation) if enabled else None
