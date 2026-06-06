"""Drift detection: compare two run summaries.

For each metric we compute a delta plus a two-sample t-test on the per-
sample scores. We flag a metric as drifted if |delta_mean| >= 0.03 OR
the t-test p-value is < 0.05. The thresholds are conservative defaults;
tune via the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from scipy import stats


@dataclass
class DriftReport:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    p_value: float
    drifted: bool


def _read_samples(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def detect(
    baseline_samples_path: Path,
    candidate_samples_path: Path,
    delta_threshold: float = 0.03,
    p_threshold: float = 0.05,
) -> list[DriftReport]:
    base = _read_samples(baseline_samples_path)
    cand = _read_samples(candidate_samples_path)
    if not base or not cand:
        logger.warning("empty baseline or candidate")
        return []

    metrics = sorted(k for k in base[0] if k not in {"qid", "question"} and not k.endswith("__why"))

    reports: list[DriftReport] = []
    for m in metrics:
        b_vals = [float(r[m]) for r in base if m in r]
        c_vals = [float(r[m]) for r in cand if m in r]
        if len(b_vals) < 2 or len(c_vals) < 2:
            continue
        bm = sum(b_vals) / len(b_vals)
        cm = sum(c_vals) / len(c_vals)
        # Welch's t-test (unequal variances)
        _, p = stats.ttest_ind(b_vals, c_vals, equal_var=False)
        delta = cm - bm
        drifted = abs(delta) >= delta_threshold or p < p_threshold
        reports.append(
            DriftReport(
                metric=m,
                baseline_mean=bm,
                candidate_mean=cm,
                delta=delta,
                p_value=float(p),
                drifted=drifted,
            )
        )
    return reports
