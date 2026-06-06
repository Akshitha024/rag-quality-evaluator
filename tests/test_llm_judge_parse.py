from __future__ import annotations

from rev.judges.llm import _parse


def test_parse_clean_json() -> None:
    s, why = _parse('{"score": 0.8, "rationale": "looks good"}')
    assert s == 0.8
    assert why == "looks good"


def test_parse_in_code_fence() -> None:
    text = 'Here you go:\n```json\n{"score": 0.5, "rationale": "partial"}\n```'
    s, why = _parse(text)
    assert s == 0.5
    assert why == "partial"


def test_parse_no_json() -> None:
    s, _ = _parse("This is good!")
    assert s == 0.0


def test_parse_missing_score() -> None:
    s, _ = _parse('{"rationale": "no score key"}')
    assert s == 0.0
