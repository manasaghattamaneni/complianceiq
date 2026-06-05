# utils/logger.py
# Structured JSON logging for the entire application
# Every event is logged with timestamp, level, and context

import logging
import json
import traceback
from datetime import datetime, timezone
from config import APP_NAME, LOG_FILE, LOG_LEVEL


class StructuredLogger:
    """
    Produces structured JSON logs.

    Why JSON logs?
    - Machine readable → easy to search in Datadog/CloudWatch
    - Consistent format → every log has same fields
    - Filterable → find all errors in seconds
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Prevent duplicate handlers if logger already exists
        if not self.logger.handlers:
            # Console handler — shows logs in terminal
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(console_handler)

            # File handler — saves logs to file
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)

    def _get_formatter(self):
        return logging.Formatter("%(message)s")

    def _build_log(self, level: str, event: str, **kwargs) -> str:
        """Build a structured JSON log entry."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app": APP_NAME,
            "level": level,
            "event": event,
            **kwargs,  # any extra fields passed in
        }
        return json.dumps(log_entry)

    def info(self, event: str, **kwargs):
        """Log an informational event."""
        self.logger.info(self._build_log("INFO", event, **kwargs))

    def warning(self, event: str, **kwargs):
        """Log a warning — something unexpected but not fatal."""
        self.logger.warning(self._build_log("WARNING", event, **kwargs))

    def error(self, event: str, exception: Exception = None, **kwargs):
        """Log an error with optional full stack trace."""
        extra = {}
        if exception:
            extra["error_type"] = type(exception).__name__
            extra["error_message"] = str(exception)
            extra["stack_trace"] = traceback.format_exc()
        self.logger.error(self._build_log("ERROR", event, **extra, **kwargs))

    def log_upload(
        self, doc_name: str, file_type: str, pages: int, chunks: int, duration_ms: float
    ):
        """Log a document upload event."""
        self.info(
            "document_uploaded",
            doc_name=doc_name,
            file_type=file_type,
            pages=pages,
            chunks=chunks,
            duration_ms=round(duration_ms, 2),
        )

    def log_query(
        self,
        question_length: int,
        num_docs: int,
        confidence: float,
        duration_ms: float,
        token_count: int,
    ):
        """Log a question/answer event."""
        self.info(
            "query_processed",
            question_length=question_length,
            num_docs_searched=num_docs,
            top_confidence_pct=round(confidence, 1),
            duration_ms=round(duration_ms, 2),
            token_count=token_count,
        )

    def log_feedback(self, feedback_type: str, question_length: int):
        """Log user feedback on an answer."""
        self.info(
            "feedback_received",
            feedback_type=feedback_type,
            question_length=question_length,
        )

    def log_error(self, event: str, exception: Exception, doc_name: str = None):
        """Log an error with context."""
        self.error(event, exception=exception, doc_name=doc_name or "unknown")


# Single shared logger instance
# Import this everywhere: from utils.logger import logger
logger = StructuredLogger(APP_NAME)
