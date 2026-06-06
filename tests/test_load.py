from __future__ import annotations

from pathlib import Path

from rev.evaluators.run import load_samples

FIX = Path(__file__).parent / "fixtures"


def test_load_synthetic_fixture() -> None:
    samples = load_samples(FIX / "synthetic.jsonl")
    assert len(samples) == 10
    s0 = samples[0]
    assert s0.qid == "q1"
    assert "Paris" in s0.answer
    assert s0.gold == "Paris"
    assert s0.citations == (0,)
    assert len(s0.contexts) == 2
