"""Graph store interface + an in-memory implementation.

The grounded graph links chunk nodes to UMLS concept nodes (chunk -MENTIONS->
concept) and concepts to each other (concept -RELATED-> concept). Graph
retrieval grounds a query to concepts, expands a bounded neighbourhood, and
ranks chunks by how many (query / neighbour) concepts they mention.

Every chunk carries provenance (source-file, collection, layer, chunk index) so
no node has a blank scope (Doc 00 Step 2 / control: legacy blank scope).

InMemoryGraphStore backs dev, tests, and the smoke; Neo4jStore (neo4j_store.py)
implements the same protocol against a real database for the full build.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Protocol


class GraphStore(Protocol):
    def add_chunk(self, chunk_id: str, text: str, **provenance) -> None: ...
    def add_concept(self, cui: str) -> None: ...
    def link_chunk_concept(self, chunk_id: str, cui: str) -> None: ...
    def link_concept_concept(self, a: str, b: str, rel: str = "related") -> None: ...
    def chunks_for_concepts(self, cuis: set[str], *, hops: int = 1, limit: int = 10) -> list[tuple[str, float]]: ...
    def text(self, chunk_id: str) -> str: ...
    def concepts_of(self, chunk_id: str) -> set[str]: ...


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.chunks: dict[str, dict] = {}
        self.chunk_concepts: dict[str, set[str]] = defaultdict(set)
        self.concept_chunks: dict[str, set[str]] = defaultdict(set)
        self.concept_edges: dict[str, set[str]] = defaultdict(set)

    def add_chunk(self, chunk_id: str, text: str, **provenance) -> None:
        self.chunks[chunk_id] = {"text": text, **provenance}

    def add_concept(self, cui: str) -> None:
        self.concept_edges.setdefault(cui, set())

    def link_chunk_concept(self, chunk_id: str, cui: str) -> None:
        self.chunk_concepts[chunk_id].add(cui)
        self.concept_chunks[cui].add(chunk_id)

    def link_concept_concept(self, a: str, b: str, rel: str = "related") -> None:
        self.concept_edges[a].add(b)
        self.concept_edges[b].add(a)

    def expand_concepts(self, cuis: set[str], hops: int) -> set[str]:
        seen, frontier = set(cuis), set(cuis)
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for c in frontier:
                nxt |= self.concept_edges.get(c, set())
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return seen

    def chunks_for_concepts(self, cuis: set[str], *, hops: int = 1, limit: int = 10) -> list[tuple[str, float]]:
        qset = set(cuis)
        expanded = self.expand_concepts(qset, hops)
        scores: dict[str, float] = defaultdict(float)
        for c in expanded:
            w = 1.0 if c in qset else 0.5  # direct query concepts weigh more than neighbours
            for ch in self.concept_chunks.get(c, set()):
                scores[ch] += w
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]

    def text(self, chunk_id: str) -> str:
        return self.chunks.get(chunk_id, {}).get("text", "")

    def concepts_of(self, chunk_id: str) -> set[str]:
        return set(self.chunk_concepts.get(chunk_id, set()))

    def signature(self) -> str:
        """Stable hash over nodes + edges — the basis of graph_hash."""
        parts = []
        for cid in sorted(self.chunks):
            parts.append("C|" + cid + "|" + ",".join(sorted(self.chunk_concepts.get(cid, set()))))
        for a in sorted(self.concept_edges):
            parts.append("E|" + a + "|" + ",".join(sorted(self.concept_edges[a])))
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    # -- persistence ----------------------------------------------------------
    # Without this the in-memory build is lost when the process exits, so the
    # graph arms are reachable only through Neo4j. A JSON snapshot lets a box
    # with no database build the graph once and retrieve against it afterwards;
    # ``graph_hash`` round-trips, so a reloaded graph is still auditable against
    # the G3 evidence that froze it.

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "chunks": self.chunks,
            "chunk_concepts": {k: sorted(v) for k, v in self.chunk_concepts.items() if v},
            "concept_edges": {k: sorted(v) for k, v in self.concept_edges.items()},
            "graph_hash": self.signature(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InMemoryGraphStore":
        store = cls()
        for cid, payload in (d.get("chunks") or {}).items():
            payload = dict(payload)
            store.add_chunk(cid, payload.pop("text", ""), **payload)
        for cid, cuis in (d.get("chunk_concepts") or {}).items():
            for cui in cuis:
                store.link_chunk_concept(cid, cui)
        for a, neighbours in (d.get("concept_edges") or {}).items():
            store.add_concept(a)
            for b in neighbours:
                store.link_concept_concept(a, b)
        return store


def save_snapshot(store: InMemoryGraphStore, path) -> str:
    """Write a JSON snapshot; returns the frozen ``graph_hash``."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = store.to_dict()
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(payload["graph_hash"])


def load_snapshot(path) -> InMemoryGraphStore:
    """Rebuild a store from :func:`save_snapshot`, verifying the hash survived."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    store = InMemoryGraphStore.from_dict(payload)
    expected = payload.get("graph_hash")
    if expected and store.signature() != expected:
        raise ValueError(
            f"graph snapshot hash mismatch: file says {expected}, reload computes {store.signature()}"
        )
    return store
