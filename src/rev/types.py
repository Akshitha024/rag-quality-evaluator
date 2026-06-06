"""Core types for the RAG evaluator.

A RAG sample = (question, generated answer, retrieved context chunks, gold
answer). The scorers each take a sample and return a 0-1 score plus a
rationale (free-form string) so the per-sample table can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sample:
    qid: str
    question: str
    answer: str
    contexts: tuple[str, ...]
    gold: str | None = None
    citations: tuple[int, ...] = ()  # context indices the answer cites


@dataclass
class MetricScore:
    name: str
    score: float
    rationale: str = ""
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class SampleResult:
    sample: Sample
    metrics: list[MetricScore]

    def get(self, metric: str) -> float:
        for m in self.metrics:
            if m.name == metric:
                return m.score
        return 0.0
