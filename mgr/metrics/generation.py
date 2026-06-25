"""Generation metrics: exact match, token-F1, MCQ accuracy.

These consume *normalized* predictions (mgr.generate.extract). EM/F1 must only
be reported after the answer-format audit passes (mgr.eval.answer_format_audit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def exact_match(pred: str | None, gold: str | None) -> int:
    if pred is None or gold is None:
        return 0
    return int(pred.strip().lower() == gold.strip().lower())


def token_f1(pred: str | None, gold: str | None) -> float:
    if pred is None or gold is None:
        return 0.0
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: dict[str, int] = {}
    gset = list(g)
    for t in p:
        if t in gset:
            common[t] = common.get(t, 0) + 1
            gset.remove(t)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p)
    recall = n_common / len(g)
    return 2 * precision * recall / (precision + recall)


@dataclass
class GenerationScores:
    n: int
    accuracy: float       # mean EM over labelled items (the MCQ/yes-no headline)
    em: float
    f1: float
    coverage: float       # fraction of items with an extractable prediction


def score(preds: list[str | None], golds: list[str | None]) -> GenerationScores:
    assert len(preds) == len(golds), "preds/golds length mismatch"
    n = len(preds)
    if n == 0:
        return GenerationScores(0, 0.0, 0.0, 0.0, 0.0)
    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [token_f1(p, g) for p, g in zip(preds, golds)]
    covered = sum(1 for p in preds if p is not None)
    return GenerationScores(
        n=n,
        accuracy=sum(ems) / n,
        em=sum(ems) / n,
        f1=sum(f1s) / n,
        coverage=covered / n,
    )
