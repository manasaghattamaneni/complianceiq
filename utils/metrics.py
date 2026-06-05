# utils/metrics.py
# Performance and usage tracking
# Tracks latency, token costs, feedback quality, error rates

import time
from dataclasses import dataclass, field
from typing import List
from utils.logger import logger


@dataclass
class QueryMetric:

    question_length: int
    duration_ms: float
    token_count: int
    confidence_pct: float


@dataclass
class SessionMetrics:
    """
    Tracks metrics for the entire user session.
    Resets when a new document is uploaded.
    """

    total_queries: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    helpful_count: int = 0
    unhelpful_count: int = 0
    error_count: int = 0
    queries: List[QueryMetric] = field(default_factory=list)

    def add_query(self, metric: QueryMetric):
        """Record a completed query."""
        self.queries.append(metric)
        self.total_queries += 1
        self.total_tokens += metric.token_count
        self.total_duration_ms += metric.duration_ms

    def sync_feedback(self, feedback_values):
        """
        Recompute feedback counts from the feedback dict (source of truth).
        Idempotent — re-clicking or toggling 👍/👎 never double-counts.
        """
        values = list(feedback_values)
        self.helpful_count = sum(1 for v in values if v == "up")
        self.unhelpful_count = sum(1 for v in values if v == "down")

    def add_error(self):
        """Record a failed query."""
        self.error_count += 1

    @property
    def avg_latency_ms(self) -> float:
        """Average response time in milliseconds."""
        if self.total_queries == 0:
            return 0.0
        return round(self.total_duration_ms / self.total_queries, 1)

    @property
    def feedback_ratio(self) -> str:
        """Helpful answers as a fraction string."""
        total = self.helpful_count + self.unhelpful_count
        if total == 0:
            return "—"
        return f"{self.helpful_count}/{total}"

    @property
    def avg_confidence(self) -> float:
        """
        Average retrieval confidence across Q&A queries only.
        One-shot tasks (gap analysis / checklist) have no single top
        confidence (recorded with question_length=0) and are excluded.
        """
        qa = [q for q in self.queries if q.question_length > 0]
        if not qa:
            return 0.0
        return round(sum(q.confidence_pct for q in qa) / len(qa), 1)

    def reset(self):
        """Reset all metrics — called when new document uploaded."""
        self.total_queries = 0
        self.total_tokens = 0
        self.total_duration_ms = 0.0
        self.helpful_count = 0
        self.unhelpful_count = 0
        self.error_count = 0
        self.queries = []
        logger.info("session_metrics_reset")


class Timer:
    
    def __enter__(self):
        self.start = time.perf_counter()
        self.duration_ms = 0.0
        return self

    def __exit__(self, *args):
        self.duration_ms = (time.perf_counter() - self.start) * 1000
