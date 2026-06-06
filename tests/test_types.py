from __future__ import annotations

from rev.types import MetricScore, Sample, SampleResult


def test_sample_is_frozen() -> None:
    s = Sample(qid="q1", question="?", answer="!", contexts=("a", "b"))
    try:
        s.qid = "q2"  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("Sample should be frozen")


def test_get_metric_returns_zero_if_missing() -> None:
    s = Sample(qid="q1", question="?", answer="!", contexts=())
    r = SampleResult(sample=s, metrics=[MetricScore(name="faithfulness", score=0.7)])
    assert r.get("faithfulness") == 0.7
    assert r.get("nonexistent") == 0.0
