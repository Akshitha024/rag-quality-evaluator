"""Orchestrator: load samples, run every metric, write per-sample + aggregate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from loguru import logger
from tqdm import tqdm

from ..types import MetricScore, Sample, SampleResult


class JudgeProto(Protocol):
    name: str

    def faithfulness(self, s: Sample) -> tuple[float, str]: ...
    def answer_relevance(self, s: Sample) -> tuple[float, str]: ...
    def citation_grounding(self, s: Sample) -> tuple[float, str]: ...


def load_samples(path: Path) -> list[Sample]:
    out: list[Sample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                Sample(
                    qid=str(obj["qid"]),
                    question=str(obj["question"]),
                    answer=str(obj["answer"]),
                    contexts=tuple(obj.get("contexts") or ()),
                    gold=obj.get("gold"),
                    citations=tuple(int(c) for c in obj.get("citations", [])),
                )
            )
    logger.info("loaded {} samples from {}", len(out), path)
    return out


def score(
    samples: list[Sample], judge: JudgeProto, heuristic_only: bool = False
) -> list[SampleResult]:
    """Run every metric on every sample. Heuristic judge has more metrics
    (context_precision, context_recall, answer_correctness) because they
    don't need an LLM; the LLM judge focuses on the harder semantic ones.
    """
    results: list[SampleResult] = []
    for s in tqdm(samples, desc=f"score:{judge.name}"):
        ms: list[MetricScore] = []
        for metric_name in ("faithfulness", "answer_relevance", "citation_grounding"):
            fn = getattr(judge, metric_name)
            v, r = fn(s)
            ms.append(MetricScore(name=metric_name, score=v, rationale=r))
        if heuristic_only:
            for extra in ("context_precision", "context_recall", "answer_correctness"):
                fn = getattr(judge, extra, None)
                if fn is None:
                    continue
                v, r = fn(s)
                ms.append(MetricScore(name=extra, score=v, rationale=r))
        results.append(SampleResult(sample=s, metrics=ms))
    return results


def write_results(out_dir: Path, run_id: str, results: list[SampleResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample = out_dir / f"{run_id}__samples.jsonl"
    summary = out_dir / f"{run_id}__summary.json"

    metric_names = sorted({m.name for r in results for m in r.metrics})
    with per_sample.open("w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "qid": r.sample.qid,
                        "question": r.sample.question,
                        **{m.name: m.score for m in r.metrics},
                        **{f"{m.name}__why": m.rationale for m in r.metrics},
                    }
                )
                + "\n"
            )

    summary_dict: dict[str, Any] = {"run_id": run_id, "n": len(results), "metrics": {}}
    for name in metric_names:
        vals = [r.get(name) for r in results if r.get(name) is not None]
        if not vals:
            continue
        summary_dict["metrics"][name] = {
            "mean": float(sum(vals) / len(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "n": len(vals),
        }
    summary.write_text(json.dumps(summary_dict, indent=2))
    logger.info("wrote {} + {}", per_sample.name, summary.name)
    return summary
