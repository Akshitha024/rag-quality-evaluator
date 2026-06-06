from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tabulate import tabulate

from ..evaluators.drift import detect
from ..evaluators.run import load_samples, score, write_results
from ..judges.heuristic import HeuristicJudge
from ..judges.llm import LLMJudge, LLMJudgeCfg
from ..viz.charts import (
    plot_correlation_heatmap,
    plot_drift_bars,
    plot_judge_calibration,
    plot_radar,
    plot_score_buckets,
    plot_violins,
)

app = typer.Typer(add_completion=False, help="rev: RAG evaluator")


@app.command("score")
def cmd_score(
    data: Annotated[Path, typer.Option(help="JSONL of samples")] = Path("data.jsonl"),
    provider: Annotated[
        str, typer.Option(help="judge: heuristic | anthropic | openai")
    ] = "heuristic",
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    run_id: Annotated[str, typer.Option(help="label this run")] = "latest",
) -> None:
    samples = load_samples(data)
    if provider == "heuristic":
        judge = HeuristicJudge()
        results = score(samples, judge, heuristic_only=True)
    else:
        cfg = LLMJudgeCfg(provider=provider)
        judge_llm = LLMJudge(cfg)
        results = score(samples, judge_llm, heuristic_only=False)
    summary_path = write_results(out_dir, run_id, results)
    summary = json.loads(summary_path.read_text())
    rows = [(name, m["mean"], m["min"], m["max"], m["n"]) for name, m in summary["metrics"].items()]
    print()
    print(
        tabulate(
            rows, headers=["metric", "mean", "min", "max", "n"], floatfmt=".3f", tablefmt="github"
        )
    )


@app.command("drift")
def cmd_drift(
    baseline: Annotated[Path, typer.Option(help="baseline samples jsonl")] = Path(
        "results/baseline__samples.jsonl"
    ),
    candidate: Annotated[Path, typer.Option(help="candidate samples jsonl")] = Path(
        "results/latest__samples.jsonl"
    ),
    out: Annotated[Path, typer.Option(help="drift report json")] = Path("results/drift.json"),
) -> None:
    reports = detect(baseline, candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in reports], indent=2))
    rows = [
        (r.metric, r.baseline_mean, r.candidate_mean, r.delta, r.p_value, r.drifted)
        for r in reports
    ]
    print()
    print(
        tabulate(
            rows,
            headers=["metric", "base", "cand", "delta", "p", "drifted?"],
            floatfmt=".3f",
            tablefmt="github",
        )
    )
    if not reports:
        logger.warning("no drift to report")


@app.command("plots")
def cmd_plots(
    samples: Annotated[Path, typer.Option(help="primary samples jsonl")] = Path(
        "results/latest__samples.jsonl"
    ),
    baseline_samples: Annotated[Path, typer.Option(help="for calibration plot")] = Path(
        "results/baseline__samples.jsonl"
    ),
    drift_json: Annotated[Path, typer.Option(help="drift json")] = Path("results/drift.json"),
    out_dir: Annotated[Path, typer.Option(help="figures dir")] = Path("results/figures"),
    calibration_metric: Annotated[
        str, typer.Option(help="metric for calibration plot")
    ] = "faithfulness",
) -> None:
    plot_radar(samples, out_dir / "radar.png")
    plot_correlation_heatmap(samples, out_dir / "correlation_heatmap.png")
    plot_violins(samples, out_dir / "violins.png")
    plot_score_buckets(samples, out_dir / "score_buckets.png")
    plot_drift_bars(drift_json, out_dir / "drift_bars.png")
    plot_judge_calibration(
        baseline_samples,
        samples,
        calibration_metric,
        out_dir / f"calibration_{calibration_metric}.png",
    )
    typer.echo(f"wrote 6 figures to {out_dir}")


if __name__ == "__main__":
    app()
