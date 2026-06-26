"""Retrieval metrics: Recall@k, Precision@k, MRR, nDCG (Metrics sheet families).

These score a ranked ``retrieved`` id list against a set of ``relevant`` (gold)
ids per query, then aggregate over a benchmark. Recall@3 is the sheet's primary
retrieval metric. Together with the generation metrics they form the substrate
for the Retrieval-Generation Decomposition (C1): retrieval gain vs generation
gain on the same items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

STD_RECALL_K = (1, 3, 5, 10)
STD_PRECISION_K = (1, 3, 5)
STD_NDCG_K = (10,)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hit = sum(1 for d in retrieved[:k] if d in rel)
    return hit / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    hit = sum(1 for d in retrieved[:k] if d in rel)
    return hit / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(retrieved[:k], start=1) if d in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalScores:
    n: int
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)


def score(
    items: Sequence[tuple[Sequence[str], Iterable[str]]],
    *,
    recall_k: Sequence[int] = STD_RECALL_K,
    precision_k: Sequence[int] = STD_PRECISION_K,
    ndcg_k: Sequence[int] = STD_NDCG_K,
) -> RetrievalScores:
    """Aggregate retrieval metrics over (retrieved_ids, relevant_ids) pairs."""
    n = len(items)
    if n == 0:
        return RetrievalScores(0)
    rec = {k: 0.0 for k in recall_k}
    prec = {k: 0.0 for k in precision_k}
    nd = {k: 0.0 for k in ndcg_k}
    mrr = 0.0
    for retrieved, relevant in items:
        relevant = list(relevant)
        for k in recall_k:
            rec[k] += recall_at_k(retrieved, relevant, k)
        for k in precision_k:
            prec[k] += precision_at_k(retrieved, relevant, k)
        for k in ndcg_k:
            nd[k] += ndcg_at_k(retrieved, relevant, k)
        mrr += reciprocal_rank(retrieved, relevant)
    return RetrievalScores(
        n=n,
        recall={k: v / n for k, v in rec.items()},
        precision={k: v / n for k, v in prec.items()},
        mrr=mrr / n,
        ndcg={k: v / n for k, v in nd.items()},
    )
