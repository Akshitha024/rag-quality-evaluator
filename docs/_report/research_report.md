---
title: "rag-quality-evaluator: faithfulness, citation grounding, and drift detection for RAG systems"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

We present `rag-quality-evaluator`, a multi-metric quality evaluator for
RAG systems with built-in drift detection across runs. The package
implements six RAG-specific metrics (faithfulness, answer relevance,
citation grounding, context precision, context recall, answer
correctness) with two judge backends (a keyless heuristic for CI and an
LLM-as-judge for headline numbers). The drift detector runs a Welch's
t-test between two run snapshots and flags any metric that moved by
≥ 0.03 absolute or with p < 0.05. We demonstrate the harness on a
10-sample synthetic fixture with a deliberately-degraded candidate run
and show the drift detector correctly flags the regression in
`answer_correctness` (delta = -0.107, p = 0.033) while leaving the
unrelated retrieval metrics untouched.

# 1. Background

RAG systems fail in several distinct ways that a single accuracy number
hides:

- **Hallucination**: the answer makes factual claims the retrieved
  context does not support.
- **Citation grounding errors**: the answer cites the wrong chunks.
- **Retrieval failures**: the right context wasn't retrieved.
- **Over-summarization or under-summarization**: the answer addresses
  only part of the question.

A practical RAG quality evaluator therefore reports a *set* of metrics,
each targeting a specific failure mode. RAGAS (Es et al., 2023) is the
standard implementation; this project follows its metric definitions
but adds a keyless heuristic judge so the suite runs in CI, plus a
drift-detection layer so the *change* across iterations becomes a first-
class artifact.

# 2. Related Work

**RAGAS** (Es et al., 2023): the reference RAG eval framework. Our
metric definitions match theirs.

**Judging LLMs as judges.** Zheng et al. (2023). We use the per-metric
mini-rubric pattern rather than one omnibus prompt.

**SelfCheckGPT** (Manakul et al., 2023): hallucination detection by
self-consistency. Not yet wired in but the API surface matches.

**Drift detection.** Standard two-sample t-tests on per-sample scores;
nothing exotic, but treating drift detection as a first-class output
of the eval pipeline is the contribution here.

# 3. Method

## 3.1 Metrics

| metric              | what it measures                                       | judge |
|---------------------|--------------------------------------------------------|-------|
| faithfulness        | does the answer's claims appear in the retrieved context? | both |
| answer_relevance    | does the answer address the question?                  | both |
| citation_grounding  | if the answer cites chunks, do those chunks back it?   | both |
| context_precision   | fraction of retrieved chunks that are actually relevant | heur |
| context_recall      | fraction of gold-answer terms covered by chunks        | heur |
| answer_correctness  | how close is the answer to the gold (cosine + overlap) | heur |

## 3.2 Heuristic judge

Sentence-transformers (BGE-small) embeddings + token overlap (Jaccard).
The metrics are coarser than the LLM judge but the relative ordering
across samples is preserved, which is what the drift detector cares
about.

## 3.3 LLM judge

Per-metric mini-rubric prompts: each judge call asks for one 0-1 score
plus a one-line rationale, returned as JSON. Smaller rubrics produce
more consistent JSON outputs at lower judge cost than one omnibus
prompt. Council mode (N judges) is implemented.

## 3.4 Drift detection

```python
from scipy.stats import ttest_ind
_, p = ttest_ind(baseline_scores, candidate_scores, equal_var=False)
delta = mean(candidate) - mean(baseline)
flagged = abs(delta) >= 0.03 OR p < 0.05
```

Welch's t-test (unequal variances) per metric. No multi-test correction
in v1; that's the obvious next addition.

# 4. Data

The in-repo synthetic fixture has 10 hand-curated Q/A items. Two
versions (`synthetic.jsonl` and `synthetic_v2.jsonl`); v2 has five
intentionally-degraded answers (off-topic, vague, factually wrong) so
the drift detector has something to find.

# 5. Evaluation Setup

Baseline + candidate runs through the heuristic judge, drift check,
then five-chart plot generation. Hardware: Apple M-series CPU.

# 6. Results

Baseline run (`synthetic.jsonl`, n=10):

| metric             |  mean | min   | max   |
|--------------------|------:|------:|------:|
| faithfulness       | 0.912 | 0.864 | 0.992 |
| citation_grounding | 0.909 | 0.864 | 0.992 |
| context_recall     | 0.854 | 0.375 | 1.000 |
| answer_relevance   | 0.811 | 0.681 | 0.875 |
| context_precision  | 0.550 | 0.000 | 1.000 |
| answer_correctness | 0.535 | 0.337 | 0.651 |

Drift (candidate `synthetic_v2.jsonl` minus baseline):

| metric             |  delta |     p  | flagged |
|--------------------|-------:|-------:|--------:|
| answer_correctness | -0.107 | 0.033  | yes (*) |
| citation_grounding | -0.041 | 0.164  | yes     |
| faithfulness       | -0.033 | 0.203  | yes     |
| answer_relevance   | -0.025 | 0.410  | no      |
| context_precision  |  0.000 | 1.000  | no      |
| context_recall     |  0.000 | 1.000  | no      |

The two `context_*` metrics correctly stayed at 0.000 delta because
the candidate fixture changes only the *answers*, not the retrieved
contexts; this is what a working drift detector should show. The
generation-side metrics moved as expected: `answer_correctness` was
the leading indicator (p < 0.05), followed by `citation_grounding`
and `faithfulness` flagged by delta threshold.

# 7. Ablations

The threshold pair (delta = 0.03, p = 0.05) is conservative; in
practice you'd tune per-metric. We chose those values because they
catch the synthetic-degradation regression without false-positives
on a re-run of the same fixture.

# 8. Discussion

The drift-detection framing is the most valuable thing in this repo.
Most RAG eval tools score a single run; the *change* in scores
across iterations is what actually informs the "should I ship this
prompt change" decision. Wrapping a standard t-test around the
per-sample scores is two functions of code; not having it is
mostly an oversight in the existing tools.

# 9. Limitations

1. **Per-metric tests are independent.** No Bonferroni / BH
   correction. With 6 metrics at p < 0.05, the family-wise false
   positive rate is ~26%.
2. **Heuristic judge is coarse.** It captures topical similarity
   well but is weak on negation. Use LLM judge for headline numbers.
3. **No bootstrap CIs.** Per-metric means are point estimates.
4. **The fixture is small (n=10).** Real RAG eval sets are
   typically 100-1000 items.

# 10. Future Work

- [ ] Bonferroni / Benjamini-Hochberg correction on drift p-values.
- [ ] Bootstrap CIs on per-metric means.
- [ ] Council judge with reported inter-judge agreement.
- [ ] Failure-category labels per sample (hallucinated /
      off-topic / under-supported / irrelevant context).
- [ ] Hooks for streaming RAG (chunked answers).

# 11. References

- Es, S., et al. (2023). *RAGAS: Automated Evaluation of Retrieval
  Augmented Generation.* arXiv:2309.15217.
- Manakul, P., et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box
  Hallucination Detection.* EMNLP.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS.

# Appendix A. Reproducibility

- Repo: `Akshitha024/rag-quality-evaluator`, MIT.
- Reproduce: `make eval && cp results/latest__samples.jsonl
  results/baseline__samples.jsonl && make eval DATA=tests/fixtures/synthetic_v2.jsonl
  && make drift && make plots`.
- Test artifacts in `docs/test_results/`.
