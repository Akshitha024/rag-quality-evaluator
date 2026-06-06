"""A no-LLM judge built from sentence-transformers + token overlap.

This is intentionally cheap so the evaluator runs without API keys and in
CI. The numbers it produces are coarser than a real LLM judge, but the
relative orderings (which sample faithfulness is higher than which) are
useful for regression detection across iterations.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..types import Sample

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_WORD = re.compile(r"\w+")


def tokens(text: str) -> set[str]:
    return {t for t in (m.group(0).lower() for m in _WORD.finditer(text)) if len(t) > 1}


def token_overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class HeuristicJudge:
    """No LLM. Scoring is cosine-similarity + token overlap."""

    name = "heuristic"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._st: SentenceTransformer | None = None

    def _embed(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        if self._st is None:
            self._st = SentenceTransformer(self.model_name)
        return self._st

    def cosine(self, a: str, b: str) -> float:
        m = self._embed()
        emb = m.encode([a, b], normalize_embeddings=True, convert_to_numpy=True)
        return float((emb[0] * emb[1]).sum())

    def faithfulness(self, s: Sample) -> tuple[float, str]:
        # answer should be supported by context: max cosine with any chunk
        if not s.contexts:
            return 0.0, "no contexts"
        sims = [self.cosine(s.answer, c) for c in s.contexts]
        best = max(sims)
        return best, f"max cosine with any chunk = {best:.3f}"

    def answer_relevance(self, s: Sample) -> tuple[float, str]:
        # answer should be on-topic for the question
        sim = self.cosine(s.answer, s.question)
        return sim, f"cosine(answer, question) = {sim:.3f}"

    def context_precision(self, s: Sample) -> tuple[float, str]:
        # fraction of contexts that have non-trivial overlap with the gold answer
        if not s.contexts or s.gold is None:
            return 0.0, "no gold or no contexts"
        relevant = sum(1 for c in s.contexts if token_overlap(c, s.gold) >= 0.1)
        prec = relevant / len(s.contexts)
        return prec, f"{relevant}/{len(s.contexts)} contexts overlap >= 0.1 with gold"

    def context_recall(self, s: Sample) -> tuple[float, str]:
        # do the contexts together cover the gold answer's terms?
        if not s.contexts or s.gold is None:
            return 0.0, "no gold or no contexts"
        gold = tokens(s.gold)
        joined = " ".join(s.contexts)
        covered = tokens(joined) & gold
        rec = len(covered) / len(gold) if gold else 0.0
        return rec, f"{len(covered)}/{len(gold)} gold tokens covered by contexts"

    def citation_grounding(self, s: Sample) -> tuple[float, str]:
        # if the answer claims citations, do they actually back the answer text?
        if not s.citations:
            # no citations: full credit (no false attribution) if answer is short
            return 1.0 if len(s.answer) < 200 else 0.5, "no explicit citations"
        sims = []
        for idx in s.citations:
            if 0 <= idx < len(s.contexts):
                sims.append(self.cosine(s.answer, s.contexts[idx]))
        if not sims:
            return 0.0, "all citation indices out of range"
        avg = sum(sims) / len(sims)
        return avg, f"avg cosine between answer and cited contexts = {avg:.3f}"

    def answer_correctness(self, s: Sample) -> tuple[float, str]:
        if s.gold is None:
            return 0.0, "no gold"
        sim = self.cosine(s.answer, s.gold)
        overlap = token_overlap(s.answer, s.gold)
        score = 0.5 * sim + 0.5 * overlap
        return score, f"0.5*cosine + 0.5*overlap = {score:.3f}"
