"""LLM-as-judge for RAG metrics.

We use a small, focused rubric per metric instead of one omnibus prompt.
Smaller rubrics give more consistent JSON outputs and reduce judge cost
(shorter outputs). Each judge call asks for one number (0-1) plus a one-
line rationale.

Provider auth comes from env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from loguru import logger

from ..types import Sample

_FAITHFULNESS = """\
You are evaluating whether a generated answer is supported by retrieved context.

Question:
{question}

Generated answer:
{answer}

Retrieved context chunks:
{contexts}

Score the answer's faithfulness to the context on a 0.0-1.0 scale.
1.0 = every factual claim is supported by some chunk.
0.0 = answer makes factual claims that contradict or are absent from the chunks.

Respond as JSON only: {{"score": <float 0-1>, "rationale": "<one short sentence>"}}
"""

_RELEVANCE = """\
Score whether the answer addresses the question.

Question: {question}
Answer:   {answer}

Respond as JSON only: {{"score": <float 0-1>, "rationale": "<one short sentence>"}}
"""

_CITATION = """\
The answer claims to be backed by these specific context chunks:
{cited_contexts}

Answer:
{answer}

Score whether the cited chunks actually back the answer on 0.0-1.0.
1.0 = every cited chunk has content the answer relies on.
0.0 = none of the cited chunks support the answer.

Respond as JSON only: {{"score": <float 0-1>, "rationale": "<one short sentence>"}}
"""


@dataclass
class LLMJudgeCfg:
    provider: str = "anthropic"  # or "openai"
    model: str = "claude-3-5-haiku-latest"
    max_tokens: int = 100


def _need(env: str) -> str:
    v = os.environ.get(env)
    if not v:
        raise RuntimeError(f"env {env} not set")
    return v


def _parse(text: str) -> tuple[float, str]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return 0.0, f"bad judge output: {text[:80]}"
    try:
        o = json.loads(m.group(0))
        return float(o["score"]), str(o.get("rationale") or "")
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0, f"bad json: {text[:80]}"


class LLMJudge:
    name = "llm"

    def __init__(self, cfg: LLMJudgeCfg | None = None) -> None:
        self.cfg = cfg or LLMJudgeCfg()

    def _call(self, prompt: str) -> str:
        if self.cfg.provider == "anthropic":
            return self._call_anthropic(prompt)
        if self.cfg.provider == "openai":
            return self._call_openai(prompt)
        raise ValueError(f"unknown provider: {self.cfg.provider}")

    def _call_anthropic(self, prompt: str) -> str:
        from anthropic import Anthropic

        a_client = Anthropic(api_key=_need("ANTHROPIC_API_KEY"))
        r = a_client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in r.content)

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        o_client = OpenAI(api_key=_need("OPENAI_API_KEY"))
        r = o_client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""

    def faithfulness(self, s: Sample) -> tuple[float, str]:
        prompt = _FAITHFULNESS.format(
            question=s.question,
            answer=s.answer,
            contexts="\n---\n".join(f"[{i}] {c}" for i, c in enumerate(s.contexts)),
        )
        try:
            return _parse(self._call(prompt))
        except Exception as e:
            logger.warning("judge failed: {}", e)
            return 0.0, f"judge error: {e}"

    def answer_relevance(self, s: Sample) -> tuple[float, str]:
        prompt = _RELEVANCE.format(question=s.question, answer=s.answer)
        try:
            return _parse(self._call(prompt))
        except Exception as e:
            return 0.0, f"judge error: {e}"

    def citation_grounding(self, s: Sample) -> tuple[float, str]:
        if not s.citations:
            return 1.0, "no explicit citations"
        cited = [s.contexts[i] for i in s.citations if 0 <= i < len(s.contexts)]
        if not cited:
            return 0.0, "all citations out of range"
        prompt = _CITATION.format(
            answer=s.answer,
            cited_contexts="\n---\n".join(
                f"[{i}] {c}" for i, c in zip(s.citations, cited, strict=False)
            ),
        )
        try:
            return _parse(self._call(prompt))
        except Exception as e:
            return 0.0, f"judge error: {e}"
