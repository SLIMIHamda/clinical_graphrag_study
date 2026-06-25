"""Retrieval interface shared by every condition.

A Retriever maps a query to a ranked candidate list plus a rendered context
block for the prompt. Conditions differ only in their retriever/fusion wiring;
the executor and prompt path are identical across arms (prompt parity).

No-RAG uses :class:`NullRetriever` (closed-book: no context, no candidates).
BM25 / dense / graph / fusion retrievers implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RetrievalResult:
    context: str | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)
    fused_scores: list[float] = field(default_factory=list)
    rerank_fired: bool = False


class Retriever(Protocol):
    def retrieve(self, query: str, *, depth_k: int = 10) -> RetrievalResult: ...


class NullRetriever:
    """Closed-book: returns no context (No-RAG condition)."""

    def retrieve(self, query: str, *, depth_k: int = 10) -> RetrievalResult:
        return RetrievalResult()
