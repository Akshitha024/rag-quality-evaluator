from __future__ import annotations

import json
from pathlib import Path

from rev.evaluators.drift import detect


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_no_drift_when_identical(tmp_path: Path) -> None:
    rows = [
        {"qid": "q1", "question": "?", "faithfulness": 0.8, "relevance": 0.7},
        {"qid": "q2", "question": "?", "faithfulness": 0.6, "relevance": 0.5},
        {"qid": "q3", "question": "?", "faithfulness": 0.7, "relevance": 0.6},
    ]
    base = tmp_path / "base.jsonl"
    cand = tmp_path / "cand.jsonl"
    _write(base, rows)
    _write(cand, rows)
    reports = detect(base, cand)
    assert all(not r.drifted for r in reports)
    assert all(abs(r.delta) < 1e-9 for r in reports)


def test_obvious_drift(tmp_path: Path) -> None:
    base = [{"qid": f"q{i}", "question": "?", "faithfulness": 0.85} for i in range(20)]
    cand = [{"qid": f"q{i}", "question": "?", "faithfulness": 0.55} for i in range(20)]
    bp = tmp_path / "b.jsonl"
    cp = tmp_path / "c.jsonl"
    _write(bp, base)
    _write(cp, cand)
    reports = detect(bp, cp)
    assert len(reports) == 1
    r = reports[0]
    assert r.metric == "faithfulness"
    assert r.drifted is True
    assert r.delta < -0.2


def test_empty_paths_returns_empty(tmp_path: Path) -> None:
    bp = tmp_path / "b.jsonl"
    cp = tmp_path / "c.jsonl"
    bp.touch()
    cp.touch()
    assert detect(bp, cp) == []
