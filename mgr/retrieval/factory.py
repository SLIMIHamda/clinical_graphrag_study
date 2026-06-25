"""Condition -> retriever wiring.

The executor is condition-agnostic; this factory maps a Conditions-sheet name to
the retriever that defines that arm. Step 1 (the H2 smoke) needs only No-RAG and
BM25; dense / graph / fusion (RRF, CA-RRF) arms register here in later steps,
behind their gates (G3 for any graph-touching condition).
"""

from __future__ import annotations

from typing import Any

from .base import NullRetriever, Retriever
from .bm25 import BM25Index, BM25Retriever


class ConditionNotWired(NotImplementedError):
    """A condition whose retriever is not implemented yet (later build step)."""


def build_retriever(condition: str, corpus_records: list[dict[str, Any]] | None = None) -> Retriever:
    if condition == "No-RAG":
        return NullRetriever()
    if condition == "BM25":
        if not corpus_records:
            raise ValueError("BM25 condition requires a corpus")
        return BM25Retriever(BM25Index.from_records(corpus_records))
    raise ConditionNotWired(
        f"condition {condition!r} has no retriever yet "
        "(dense/graph/fusion arrive in build steps 2-5)"
    )
