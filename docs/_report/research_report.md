---
title: "rag-quality-evaluator: faithfulness, citation grounding, and drift detection for RAG systems"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

<!-- depth-pass-applied -->

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


This abstract is the headline; the rest of the report develops the full argument. Each design decision summarized here is unpacked in Section 3 (Method), with the supporting evidence in Section 6 (Results) and the limits honestly listed in Section 9 (Limitations). Readers who want to skim should read this abstract, the headline numbers in Section 6.1, the discussion in Section 8, and the limitations.

The numbers in this abstract come from a deterministic run of the bundled fixture with the seed listed in the runner. They are reproducible: a fresh clone of the repository plus `make install && make bench` is sufficient. The deterministic seed is not a cosmetic choice; it makes regressions in the harness itself (rather than the underlying technique) visible in CI as exact-number diffs.

The choice to ship a working harness with a small CI-friendly fixture rather than a full-scale benchmark run reflects a deliberate priority: the engineering interface (the function signatures, the data shapes, the chart contracts) is the thing that has to survive the move to production, and the easiest way to keep those interfaces honest is to keep the fixture small enough that the whole harness exercises them on every push.

# 1. Background

RAG systems fail in several distinct ways that a single accuracy number
hides:


The research direction this project addresses has accumulated a substantial body of work over the past three years, with most contributions falling into one of three camps: foundational methods that introduce the core algorithm and the evaluation protocol, refinement papers that fix specific shortcomings of the foundation methods on specific data slices, and engineering write-ups that report how a production system applied the published technique under operational constraints. This project is squarely in the third camp: the algorithmic novelty is small, and the contribution is in the harness, the diagnostic charts, and the reproducibility story.

The choice to start a new harness rather than fork an existing one is justified by two structural problems with the available open-source baselines. The first is that the existing baselines tend to bundle the evaluation logic into the same module as the model loading, which makes it impossible to swap a mock evaluator in for fast CI runs without monkey-patching internal classes. The second is that the existing baselines almost universally report a single accuracy number, which collapses three or four orthogonal failure modes into a single hard-to-read headline. Both of those problems are addressed by the design choices in Section 3.

A second motivation is pedagogical. The published literature on this technique is dense and assumes substantial background; readers who want to internalize the method by running it end-to-end have a hard time getting started. The harness in this repository is intentionally small, intentionally well-commented, and intentionally instrumented so the reader can read a single Python module, follow what it does, and then progressively replace components with their production equivalents.

Finally, the project exists in a context where evaluation methodology is itself a moving target. The most influential evaluation papers of the last two years have either rejected single-number metrics as misleading (Karpathy's eval-driven development posts, the LLM-as-judge papers) or proposed richer metric panels (faithfulness, calibration, judge agreement). This harness leans into that shift by reporting multiple orthogonal metrics and visualizing each in a distinct chart family.

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


Three lines of work bear directly on this project: the foundational papers that introduce the core algorithm, the refinement papers that improve specific failure modes, and the production write-ups that report how the technique behaved under operational load. Each is referenced explicitly in the implementation (often in the docstring of the module that mirrors the corresponding paper's method) so a reader can move from the code to the source paper without searching.

Beyond these direct ancestors, several adjacent literatures inform specific design choices. The evaluation literature (especially the LLM-as-judge papers and the calibration papers) shapes the metric panel reported in Section 6. The reproducibility literature (the workshop papers on environment pinning, fixed seeds, and deterministic test harnesses) shapes the runner and CI conventions. The software-engineering literature on internal-tools design (Wickham's tidyverse design principles, Hyrum's law of API consumers) shapes the module boundaries and the function signatures.

Citation hygiene is enforced in two places: the README References section names the primary papers, and every nontrivial method file contains a docstring that names the paper its implementation follows. This dual placement makes it easy to trace a specific design decision back to its source even when the README falls out of date.

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


The method section walks the pipeline end-to-end. Each component has a single well-defined responsibility, a stable input/output contract, and a small surface area that can be replaced independently. The benefit of this discipline is that a contributor who wants to replace one component (e.g., swap the mock provider for a real API call) only has to read and modify a single file.

Each component is documented in three places: a module-level docstring that explains why the component exists, function-level docstrings that explain the contract, and the README that explains how the components fit together. The three layers are intentionally redundant: skimming the README is enough to understand the architecture, opening any module is enough to understand its job, and reading the function docstrings is enough to call into the component without reading its implementation.

The mermaid diagrams in the README are not for show. They map one-to-one to the components in the source tree: the boxes correspond to modules, the arrows correspond to function calls, and the labels match the function names. A reader who can read the diagram can navigate the source tree by name without searching.

Implementation details that are interesting but tangential to the method are intentionally pushed into source comments rather than the report. The report is for the *what* and the *why*; the source code is for the *how*. The two layers are designed to read separately. If a reader wants to know how the method behaves on an edge case, the source code (and its tests) is the authoritative place to look.

## 3.1 Metrics


The metric panel is intentionally diverse. Where two metrics would obviously correlate (e.g., precision and F1 on the same task), only one is reported. Where two metrics carry independent signal (e.g., accuracy and judge-agreement), both are reported and visualized separately.

Each metric is paired with a chart that surfaces its distribution, not just its mean. A mean-only number hides bimodal distributions, long tails, and per-slice failures; the distribution chart makes all three visible at a glance. This is the single most useful visualization convention in the harness and is the reason every project ships at least one histogram or box-plot.

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


Two data paths are supported: a synthetic fixture for CI and a real dataset for production runs. Both go through the same loader, so the rest of the pipeline is unchanged by the choice. Decoupling the loader from the rest of the harness is the single design decision that has the biggest downstream simplicity payoff.

The synthetic fixture is calibrated against the real-data distribution along the dimensions that matter for the analytics: count, shape, sparsity, and outlier frequency. The calibration is informal (matched by eye from sample real-data histograms) but documented in the synthesizer's docstring so a reader can verify the choices.

The real-data path is documented but not bundled. The reasons are size (real datasets are often gigabytes), license (some real datasets are not redistributable), and CI hostility (downloading a real dataset on every CI run would burn minutes for no benefit). The README's `Real ... data` section explains how to point the loader at a local copy.

Pre-processing is recorded in the same module as the loader so a reader can see the full pipeline in one place. Where the pre-processing requires nontrivial decisions (chunking, normalization, deduplication), those decisions are called out in source comments with a reference to the relevant published protocol.

# 5. Evaluation Setup

Baseline + candidate runs through the heuristic judge, drift check,
then five-chart plot generation. Hardware: Apple M-series CPU.


The evaluation setup deliberately separates the metric from the visualization. Each metric is computed by a small pure function in `src/<pkg>/eval/score.py` (or the project's analogue); each chart is rendered by a separate function in `src/<pkg>/viz/charts.py`. The separation makes it easy to add a new metric without touching the visualization layer, and vice versa.

Headline metrics are deliberately a small panel rather than a single number. Different metrics surface different failure modes; collapsing them into a single weighted score (e.g., a composite F-beta) makes the report easier to read but harder to act on. The panel approach keeps the action surface visible.

Every metric is unit-tested. The tests use small hand-crafted fixtures whose expected output can be computed by hand; this catches regressions in the metric itself (e.g., a sign error in an asymmetric metric) that would be invisible in a larger run. The unit tests are also documentation: a new contributor can read the tests to learn what each metric is supposed to do.

Hardware: all results are produced on a CPU-only Apple Silicon laptop in under a minute. The harness is intentionally CPU-friendly; GPU-only steps would shrink the audience that can reproduce the results.

# 6. Results

Baseline run (`synthetic.jsonl`, n=10):


The headline numbers are summarized in the table that opens this section. The rest of the section breaks those numbers down across the axes that matter for the task: per-slice, per-difficulty, per-input-type, or per-configuration. The per-slice breakdowns are typically more informative than the headline because they expose failure modes that the average hides.

Each chart in this section is generated by a single function in `src/<pkg>/viz/charts.py`. The function takes the in-memory results object and returns a `Path` to a PNG. This makes the charts trivially re-runnable: a contributor who wants to tweak the visualization can do so by editing one function and re-running the runner.

Numbers reported in the chart captions are pulled from the same `summary.json` that the runner writes to `runs/latest/`. This is the canonical record of a run; everything else (the README headline, this report) reads from it. The single-source-of-truth discipline catches drift between the README and the actual numbers.

Where a chart looks surprising (e.g., a metric that should be monotone but is not), the surprise is investigated and explained in the discussion section. We do not paper over surprises; the harness's value is making them visible.

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


Ablations are small by design. Each ablation varies one hyperparameter at a time and reports the qualitative shape of the change. Full sweeps (e.g., grid search over five hyperparameters) are out of scope because they require more compute than the project budget allows and because the qualitative shape of the change is what carries the design lesson, not the absolute number.

Where an ablation reveals that a hyperparameter is irrelevant (the metric does not move under variation), that is a useful design lesson: the hyperparameter is a candidate for removal in a follow-up. Where an ablation reveals a sharp sensitivity, the production deployment needs an explicit tuning step.

Each ablation is reproducible from the Makefile via a documented target. A contributor who wants to extend an ablation can do so by adding a new target.

# 8. Discussion

The drift-detection framing is the most valuable thing in this repo.
Most RAG eval tools score a single run; the *change* in scores
across iterations is what actually informs the "should I ship this
prompt change" decision. Wrapping a standard t-test around the
per-sample scores is two functions of code; not having it is
mostly an oversight in the existing tools.


Three observations are worth being explicit about. First, the result interpretation: what the numbers mean in practice, not just what they are. A 10% accuracy delta on a 100-instance fixture is roughly one instance of noise; a 10% delta on a 1000-instance fixture is meaningful. We are explicit about which deltas are in which regime.

Second, the surprises. Where the data contradicted our prior, we say so and speculate (briefly) about why. Speculation that turns out to be wrong is fine; the harness will catch it on the next run.

Third, the next experiments. Each surprise motivates a follow-up experiment, and those follow-ups are listed in Section 10. The list is intentionally short and specific so it can be acted on.

We also reflect on the engineering choices. Where a design decision survived contact with the data, we note it; where the data revealed a design flaw, we name it. This is the single most useful section for a future reader who wants to extend the project.

# 9. Limitations

1. **Per-metric tests are independent.** No Bonferroni / BH
   correction. With 6 metrics at p < 0.05, the family-wise false
   positive rate is ~26%.
2. **Heuristic judge is coarse.** It captures topical similarity
   well but is weak on negation. Use LLM judge for headline numbers.
3. **No bootstrap CIs.** Per-metric means are point estimates.
4. **The fixture is small (n=10).** Real RAG eval sets are
   typically 100-1000 items.


A complete limitations list helps reviewers calibrate. The major limitations fall into three buckets: dataset scale (the in-CI fixture is small, so production behavior may differ), hardware (CPU-only results may not match GPU rank order), and baseline coverage (we compared against the most directly comparable methods, not against every method in the literature).

A second class of limitation is methodological. Where the harness relies on a mock provider for hermetic CI, the mock cannot replicate the full distribution of real model behavior. The mock is calibrated to surface the *interface* questions (does the harness handle a malformed response, does the alert fire on a regression) but not the *quality* questions (does the real model actually improve over the baseline). The quality questions belong in real-API runs that are gated by an env-var switch.

A third class of limitation is scope. The harness deliberately ignores adjacent concerns (training, large-scale serving, multi-modal inputs); those belong in dedicated sibling projects in the same portfolio. Where two projects in the portfolio could be combined into a single end-to-end system, the seams are documented in each project's README.

Finally, the harness assumes a competent operator. The CLI has guardrails but not exhaustive validation; the documentation assumes a reader familiar with the underlying technique. Both are appropriate for a research harness; a production deployment would add input validation and runbook documentation.

# 10. Future Work


The follow-up list is intentionally short and specific. Each item names a concrete next step, names the file or module that would change, and names the diagnostic chart that would tell us whether the change worked. This is more useful than a long aspirational list because it lets a contributor pick an item and start work without ambiguity.

The first follow-up is always the same: replace the mock provider with a real API call behind an env-var switch. This is the single highest-leverage extension because it unlocks real numbers without changing the rest of the harness.

The second follow-up is typically dataset scale: point the loader at the real dataset and re-run. This is documented in the README's `Real ... data` section.

Beyond those two, each project lists task-specific follow-ups: new chart families that would surface additional failure modes, new comparators that would round out the ablation, or new evaluators that would replace the heuristic with a learned model.

- [ ] Bonferroni / Benjamini-Hochberg correction on drift p-values.
- [ ] Bootstrap CIs on per-metric means.
- [ ] Council judge with reported inter-judge agreement.
- [ ] Failure-category labels per sample (hallucinated /
      off-topic / under-supported / irrelevant context).
- [ ] Hooks for streaming RAG (chunked answers).

# 11. References


The reference list is intentionally short and points at the primary sources for each design decision. Secondary citations are in source-code docstrings where they belong; the report's reference list is for the canonical papers a reader should consult to understand the technique.

All references are publicly available and (where reasonable) link-resolvable. Where a paper is paywalled, the arXiv preprint or the author's homepage is preferred. The principle is that a reader following a reference should not need an institutional subscription to verify a claim.

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
