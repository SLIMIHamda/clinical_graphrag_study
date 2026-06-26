"""Condition -> retriever wiring.

The executor is condition-agnostic; this factory maps a Conditions-sheet name to
the retriever that defines that arm. Step 1 (the H2 smoke) needs only No-RAG and
BM25; dense / graph / fusion (RRF, CA-RRF) arms register here in later steps,
behind their gates (G3 for any graph-touching condition).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .base import NullRetriever, Retriever
from .bm25 import BM25Index, BM25Retriever
from .dense import DenseIndex, DenseRetriever


class ConditionNotWired(NotImplementedError):
    """A condition whose retriever is not implemented yet (later build step)."""


def build_retriever(
    condition: str,
    corpus_records: list[dict[str, Any]] | None = None,
    *,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    embeddings: np.ndarray | None = None,
    doc_ids: Sequence[str] | None = None,
    passages: dict[str, str] | None = None,
) -> Retriever:
    if condition == "No-RAG":
        return NullRetriever()
    if condition == "BM25":
        if not corpus_records:
            raise ValueError("BM25 condition requires a corpus")
        return BM25Retriever(BM25Index.from_records(corpus_records))
    if condition == "Dense-MedCPT":
        if embedder is None:
            raise ValueError("Dense-MedCPT requires an embedder (for the query)")
        if embeddings is not None:
            # precomputed path: embed the corpus once offline, reuse forever
            if doc_ids is None or passages is None:
                raise ValueError("precomputed dense index needs doc_ids and passages")
            index = DenseIndex.from_embeddings(doc_ids, embeddings)
        elif corpus_records:
            index = DenseIndex.from_corpus(corpus_records, embedder)
            passages = passages or {str(r["id"]): str(r["text"]) for r in corpus_records}
        else:
            raise ValueError("Dense-MedCPT needs precomputed embeddings or a corpus")
        return DenseRetriever(index, embedder, passages)
    raise ConditionNotWired(
        f"condition {condition!r} has no retriever yet "
        "(graph/fusion arrive in build steps 2-5)"
    )
