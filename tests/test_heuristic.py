from __future__ import annotations

from rev.judges.heuristic import token_overlap, tokens
from rev.types import Sample


def test_tokens_drops_short_and_lowercases() -> None:
    assert tokens("Hello a world 12") == {"hello", "world", "12"}


def test_overlap_basic() -> None:
    assert token_overlap("the cat sat", "the cat") == 2 / 3


def test_overlap_empty_returns_zero() -> None:
    assert token_overlap("", "anything") == 0.0


def test_overlap_no_intersection() -> None:
    assert token_overlap("alpha beta", "gamma delta") == 0.0


def test_sample_constructs_with_citations() -> None:
    # the heuristic judge methods themselves require model loading and are
    # gated behind 'slow'; the pure-logic helpers above are the fast tests
    s = Sample(qid="q", question="?", answer="a", contexts=("c1", "c2"), citations=(0,))
    assert s.citations == (0,)
    assert s.contexts == ("c1", "c2")
