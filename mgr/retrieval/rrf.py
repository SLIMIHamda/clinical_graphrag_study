"""Reciprocal Rank Fusion (RRF) — the field-default fusion (Doc 1 section 4.2).

RRF(d) = sum over lists l of  w_l / (k + rank_l(d)),  rank 1-based, best first.
A document missing from a list contributes nothing for that list. The constant
``k`` damps the influence of low ranks; 60 is the conventional default and is the
same value reused by CA-RRF (the only manipulated factor there is the extra
concept list, not k).

Ties are broken deterministically by doc id so fused rankings are reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of doc ids into one (doc_id, score) ranking, desc."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match the number of ranked lists")
    scores: dict[str, float] = defaultdict(float)
    for w, lst in zip(weights, ranked_lists):
        for rank, doc in enumerate(lst, start=1):
            scores[doc] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def fused_order(ranked_lists: Sequence[Sequence[str]], *, k: int = 60, weights=None) -> list[str]:
    """Convenience: just the fused doc id order."""
    return [doc for doc, _ in reciprocal_rank_fusion(ranked_lists, k=k, weights=weights)]


def ranks_of(fused: Iterable[tuple[str, float]]) -> dict[str, int]:
    """Map doc id -> 1-based rank in a fused result."""
    return {doc: i + 1 for i, (doc, _) in enumerate(fused)}
