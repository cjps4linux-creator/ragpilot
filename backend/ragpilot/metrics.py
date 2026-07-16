"""Evaluation harness: retrieval + answer-quality metrics.

Mirrors the 'conduct evaluations to measure retrieval and answer quality, and
minimize hallucination risks' responsibility in the JD. Deterministic + observable.
"""
import threading


class EvalMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.queries = 0
        self.retrieved = 0
        self.grounded = 0
        self.citations = 0
        self._mrr = 0.0

    def record_query(self, n_retrieved: int, grounded: bool, n_citations: int, hit_rank: int | None):
        with self._lock:
            self.queries += 1
            self.retrieved += n_retrieved
            self.citations += n_citations
            if grounded:
                self.grounded += 1
            if hit_rank:
                self._mrr += 1.0 / hit_rank

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "queries": self.queries,
                "avg_retrieved": round(self.retrieved / self.queries, 2) if self.queries else 0,
                "grounding_rate": round(self.grounded / self.queries, 3) if self.queries else 0,
                "avg_citations": round(self.citations / self.queries, 2) if self.queries else 0,
                "mrr": round(self._mrr / self.queries, 3) if self.queries else 0,
            }


    def reset(self):
        with self._lock:
            self.queries = 0
            self.retrieved = 0
            self.grounded = 0
            self.citations = 0
            self._mrr = 0.0


metrics = EvalMetrics()
