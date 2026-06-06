"""Six distinct charts for RAG evaluation results.

Different vocabulary from project #4 (no nDCG curves or speed scatter)
because the eval target is different. Here we show: a radar of mean scores
across metrics, a correlation heatmap to see which metrics measure the
same thing, a calibration scatter for judge agreement, violins for per-
sample distribution, a stacked bar of score-bucket counts per metric, and
a drift report bar chart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return sorted(k for k in rows[0] if k not in {"qid", "question"} and not k.endswith("__why"))


# 1. Radar / spider of mean per-metric scores
def plot_radar(samples_path: Path, out: Path) -> Path:
    rows = _read_jsonl(samples_path)
    metrics = _metric_columns(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        out.write_bytes(b"")
        return out
    means = [float(np.mean([r[m] for r in rows if m in r])) for m in metrics]
    # close the loop
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    values = means + means[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, marker="o", linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Mean score per metric", pad=20)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 2. Correlation heatmap across metrics
def plot_correlation_heatmap(samples_path: Path, out: Path) -> Path:
    rows = _read_jsonl(samples_path)
    metrics = _metric_columns(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(metrics) < 2:
        out.write_bytes(b"")
        return out
    mat = np.array([[float(r.get(m, 0)) for m in metrics] for r in rows], dtype=np.float64)
    # row-wise std == 0 breaks correlation; jitter
    if mat.std() == 0:
        out.write_bytes(b"")
        return out
    corr = np.corrcoef(mat.T)  # (metrics, metrics)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(metrics)), max(4, 0.6 * len(metrics))))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=9)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(corr[i, j]) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Inter-metric correlation")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 3. Calibration: heuristic-judge score vs LLM-judge score per sample
def plot_judge_calibration(
    heuristic_samples: Path, llm_samples: Path, metric: str, out: Path
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not heuristic_samples.exists() or not llm_samples.exists():
        out.write_bytes(b"")
        return out
    heur = {r["qid"]: r for r in _read_jsonl(heuristic_samples)}
    llm = {r["qid"]: r for r in _read_jsonl(llm_samples)}
    common = sorted(set(heur) & set(llm))
    if not common:
        out.write_bytes(b"")
        return out
    xs = [float(heur[q][metric]) for q in common if metric in heur[q] and metric in llm[q]]
    ys = [float(llm[q][metric]) for q in common if metric in heur[q] and metric in llm[q]]
    if not xs:
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, alpha=0.6, s=40)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="y = x")
    # add Pearson r
    if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}\nn = {len(xs)}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8},
        )
    ax.set_xlabel(f"heuristic judge: {metric}")
    ax.set_ylabel(f"LLM judge: {metric}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Judge agreement on {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 4. Per-sample violin of each metric
def plot_violins(samples_path: Path, out: Path) -> Path:
    rows = _read_jsonl(samples_path)
    metrics = _metric_columns(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        out.write_bytes(b"")
        return out
    data = [[float(r[m]) for r in rows if m in r] for m in metrics]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(metrics)), 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
    bodies: Any = parts.get("bodies") or []
    for pc in bodies:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Per-sample score distribution by metric")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 5. Stacked bar of score buckets per metric (how many samples fall in each
#    [0,.25), [.25,.5), [.5,.75), [.75,1] bucket)
def plot_score_buckets(samples_path: Path, out: Path) -> Path:
    rows = _read_jsonl(samples_path)
    metrics = _metric_columns(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        out.write_bytes(b"")
        return out
    bucket_labels = ["poor [0, .25)", "weak [.25, .5)", "ok [.5, .75)", "strong [.75, 1]"]
    counts = np.zeros((len(metrics), 4))
    for i, m in enumerate(metrics):
        for r in rows:
            v = float(r.get(m, 0))
            b = 3 if v >= 0.75 else (2 if v >= 0.5 else (1 if v >= 0.25 else 0))
            counts[i, b] += 1
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(metrics)), 5))
    bottoms = np.zeros(len(metrics))
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c"]
    for j, (label, color) in enumerate(zip(bucket_labels, colors, strict=True)):
        ax.bar(metrics, counts[:, j], bottom=bottoms, label=label, color=color, width=0.7)
        bottoms += counts[:, j]
    ax.set_ylabel("# samples")
    ax.set_title("Score-bucket distribution by metric")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 6. Drift report bar chart with significance asterisks
def plot_drift_bars(drift_json_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not drift_json_path.exists():
        out.write_bytes(b"")
        return out
    reports = json.loads(drift_json_path.read_text())
    if not reports:
        out.write_bytes(b"")
        return out
    metrics = [r["metric"] for r in reports]
    deltas = [r["delta"] for r in reports]
    pvals = [r["p_value"] for r in reports]
    drifted = [r["drifted"] for r in reports]
    colors = ["#d62728" if d else "#7f7f7f" for d in drifted]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(metrics)), 4.5))
    bars = ax.bar(metrics, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(0.03, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-0.03, color="gray", linestyle=":", alpha=0.5)
    for bar, p in zip(bars, pvals, strict=True):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""
        if star:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.005 if bar.get_height() >= 0 else -0.02),
                star,
                ha="center",
                fontsize=10,
            )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("candidate - baseline (mean score)")
    ax.set_title("Per-metric drift (red = flagged, * = p<.05)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
