"""Grounding metrics — real RAGAS (faithfulness, answer relevance, context
precision/recall), Doc 1 section 5.3.

These are the "real grounding metrics" the journal upgrade demands (the thesis
used a proxy). Each is an LLM-judged quantity, so the judge is an injected
protocol: the NIM judge at runtime, a deterministic fake in tests. Answer
relevance is reported beside F1 as a first-class finding (control for the
brevity/relevance trade-off, Doc 1 section 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class GroundingJudge(Protocol):
    def decompose(self, text: str) -> list[str]:
        """Break an answer/reference into atomic claims."""
        ...

    def entails(self, hypothesis: str, premise: str) -> bool:
        """Does the premise (context) support the hypothesis (a claim)?"""
        ...

    def relevant(self, question: str, passage: str) -> bool:
        """Is a retrieved passage relevant to answering the question?"""
        ...

    def relevance(self, question: str, answer: str) -> float:
        """How relevant is the answer to the question, in [0, 1]?"""
        ...


def faithfulness(answer: str, contexts: Sequence[str], judge: GroundingJudge) -> float:
    """Fraction of answer claims supported by the retrieved context."""
    claims = judge.decompose(answer)
    if not claims:
        return 1.0  # nothing asserted -> nothing unsupported
    premise = "\n".join(contexts)
    supported = sum(1 for c in claims if judge.entails(c, premise))
    return supported / len(claims)


def answer_relevance(question: str, answer: str, judge: GroundingJudge) -> float:
    return max(0.0, min(1.0, judge.relevance(question, answer)))


def context_precision(question: str, contexts: Sequence[str], judge: GroundingJudge) -> float:
    """RAGAS-style precision: mean precision@k over the relevant-context ranks."""
    rels = [judge.relevant(question, c) for c in contexts]
    n_rel = sum(rels)
    if n_rel == 0:
        return 0.0
    num, hits = 0.0, 0
    for k, r in enumerate(rels, start=1):
        if r:
            hits += 1
            num += hits / k
    return num / n_rel


def context_recall(reference: str, contexts: Sequence[str], judge: GroundingJudge) -> float:
    """Fraction of reference (gold) claims attributable to the context."""
    claims = judge.decompose(reference)
    if not claims:
        return 1.0
    premise = "\n".join(contexts)
    covered = sum(1 for c in claims if judge.entails(c, premise))
    return covered / len(claims)


@dataclass
class GroundingScores:
    n: int
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float


@dataclass
class GroundingItem:
    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None  # gold answer, for recall


def score(items: Sequence[GroundingItem], judge: GroundingJudge) -> GroundingScores:
    n = len(items)
    if n == 0:
        return GroundingScores(0, 0.0, 0.0, 0.0, 0.0)
    f = ar = cp = cr = 0.0
    for it in items:
        f += faithfulness(it.answer, it.contexts, judge)
        ar += answer_relevance(it.question, it.answer, judge)
        cp += context_precision(it.question, it.contexts, judge)
        cr += context_recall(it.reference or "", it.contexts, judge)
    return GroundingScores(
        n=n,
        faithfulness=f / n,
        answer_relevance=ar / n,
        context_precision=cp / n,
        context_recall=cr / n,
    )
