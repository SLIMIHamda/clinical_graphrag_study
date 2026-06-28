"""Graph retriever — structural selection via the grounded graph.

The Graph-only baseline (and the graph component of the hybrid arms): ground the
query to UMLS concepts, expand a bounded neighbourhood in the graph, and return
the chunks those concepts mention, ranked by concept connectivity. Compact
context, no dense/lexical text matching — this isolates what graph *structure*
contributes (Doc 1: graph as selection/enrichment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mgr.graph.store import GraphStore

from .base import RetrievalResult, render_context


@dataclass
class GraphRetriever:
    store: GraphStore
    query_concepts_fn: Callable[[str], set[str]]
    hops: int = 1
    max_context_chars: int = 8000

    def retrieve(self, query: str, *, depth_k: int = 10) -> RetrievalResult:
        cuis = set(self.query_concepts_fn(query))
        hits = self.store.chunks_for_concepts(cuis, hops=self.hops, limit=depth_k) if cuis else []
        ids = [h[0] for h in hits]
        passages = {cid: self.store.text(cid) for cid in ids}
        return RetrievalResult(
            context=render_context(ids, passages, self.max_context_chars),
            retrieved_ids=ids,
            ranks={did: r + 1 for r, did in enumerate(ids)},
            fused_scores=[h[1] for h in hits],
            rerank_fired=False,
        )
