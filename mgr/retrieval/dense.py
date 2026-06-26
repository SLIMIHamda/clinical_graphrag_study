"""Dense retrieval (MedCPT) — biomedical embedding retrieval.

The index is decoupled from how vectors are produced: it accepts a precomputed
corpus embedding matrix (the cost-saving path — embed the corpus once, reuse
forever) OR builds one from an embedder. Only the *query* is embedded at search
time, which is a single cheap call per question.

Search is exact (flat) cosine similarity — this is also the ``noVecIndex``
ablation baseline (flat dense vs an ANN vector index, Doc 1 section 6). An ANN
(FAISS/HNSW) backend implements the same interface in a later step; swapping it
in is the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .base import RetrievalResult, render_context

# An embedder maps a batch of texts to an (n, dim) float array.
Embedder = Callable[[list[str]], np.ndarray]


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class DenseIndex:
    """Flat (exact) cosine-similarity index over corpus embeddings."""

    def __init__(self, doc_ids: Sequence[str], matrix: np.ndarray):
        self.doc_ids = list(doc_ids)
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != len(self.doc_ids):
            raise ValueError("matrix must be (n_docs, dim) aligned with doc_ids")
        self.matrix = _l2_normalize(m)
        self.dim = m.shape[1]

    @classmethod
    def from_embeddings(cls, doc_ids: Sequence[str], matrix: np.ndarray) -> "DenseIndex":
        """Build from a *precomputed* embedding matrix (no embedding cost)."""
        return cls(doc_ids, matrix)

    @classmethod
    def from_corpus(
        cls,
        records: list[dict],
        embedder: Embedder,
        *,
        id_key: str = "id",
        text_key: str = "text",
        batch_size: int = 256,
    ) -> "DenseIndex":
        """Embed a corpus once with ``embedder`` (the one-time build path)."""
        ids = [str(r[id_key]) for r in records]
        texts = [str(r[text_key]) for r in records]
        chunks = []
        for i in range(0, len(texts), batch_size):
            chunks.append(np.asarray(embedder(texts[i : i + batch_size]), dtype=float))
        matrix = np.vstack(chunks) if chunks else np.zeros((0, 0))
        return cls(ids, matrix)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        q = np.asarray(query_vec, dtype=float).ravel()
        n = np.linalg.norm(q)
        if n:
            q = q / n
        sims = self.matrix @ q
        k = min(top_k, len(self.doc_ids))
        # top-k by similarity, then stable id tie-break
        order = sorted(range(len(self.doc_ids)), key=lambda i: (-sims[i], self.doc_ids[i]))[:k]
        return [(self.doc_ids[i], float(sims[i])) for i in order]


@dataclass
class DenseRetriever:
    index: DenseIndex
    embedder: Embedder
    passages: dict[str, str]
    max_context_chars: int = 8000

    def retrieve(self, query: str, *, depth_k: int = 10) -> RetrievalResult:
        qv = np.asarray(self.embedder([query]), dtype=float)[0]
        hits = self.index.search(qv, depth_k)
        ids = [h[0] for h in hits]
        return RetrievalResult(
            context=render_context(ids, self.passages, self.max_context_chars),
            retrieved_ids=ids,
            ranks={did: r + 1 for r, did in enumerate(ids)},
            fused_scores=[h[1] for h in hits],
            rerank_fired=False,
        )
