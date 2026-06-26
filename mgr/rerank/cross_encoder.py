"""Cross-encoder reranker — the expensive step CARe gates.

At full scale this wraps a 1B reranker served on the NIM free tier (Doc 00
section 5). The interface is injectable so the CARe policy and the executor can
be tested without a model: a scorer maps (query, passage) -> relevance, and the
reranker reorders the candidate window by that score.

Cost model: reranking is O(c) forward passes per query; CARe reduces the
*expected* cost to E[g] * c by only firing on queries the gate selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class Scorer(Protocol):
    def __call__(self, query: str, passage: str) -> float: ...


@dataclass
class CrossEncoderReranker:
    scorer: Scorer

    def rerank(self, query: str, candidate_ids: list[str], passages: dict[str, str]) -> list[str]:
        """Reorder candidates by cross-encoder relevance (desc), id-tiebroken."""
        scored = [(cid, self.scorer(query, passages.get(cid, ""))) for cid in candidate_ids]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in scored]


def constant_scorer(value: float = 0.0) -> Callable[[str, str], float]:
    """A no-op scorer (keeps input order via stable tie-break) for tests."""
    return lambda _q, _p: value
