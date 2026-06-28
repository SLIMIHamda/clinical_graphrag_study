"""Grounded graph build (Doc 00 Step 2 → gate G3).

For each chunk: store the chunk node with provenance, extract entity mentions,
ground them to UMLS CUIs, and link chunk -> concept. Optional concept-concept
relations add structure for multi-hop traversal. The frozen graph_hash goes into
every graph-touching run-record.

Extraction (NIM) and linking (UMLS) are injected, so the build logic is tested
with fakes and is independent of the model/UMLS data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .store import GraphStore
from .umls import UMLSLinker


class Extractor(Protocol):
    def extract(self, text: str) -> list[str]: ...


@dataclass
class BuildReport:
    n_chunks: int
    n_concepts: int
    n_links: int
    graph_hash: str


def build_graph(
    chunks: Iterable[dict],
    store: GraphStore,
    extractor: Extractor,
    linker: UMLSLinker,
    *,
    concept_relations: Iterable[tuple[str, str]] | None = None,
) -> BuildReport:
    """Populate ``store`` from chunks. Each chunk dict: {id, text, **provenance}."""
    concepts: set[str] = set()
    n_links = 0
    n_chunks = 0
    for ch in chunks:
        n_chunks += 1
        cid = str(ch["id"])
        prov = {k: v for k, v in ch.items() if k not in {"id", "text"}}
        store.add_chunk(cid, str(ch["text"]), **prov)
        for cui in linker.concepts(extractor.extract(str(ch["text"]))):
            store.add_concept(cui)
            store.link_chunk_concept(cid, cui)
            concepts.add(cui)
            n_links += 1

    for a, b in concept_relations or []:
        store.link_concept_concept(a, b)

    graph_hash = store.signature() if hasattr(store, "signature") else ""
    return BuildReport(n_chunks=n_chunks, n_concepts=len(concepts), n_links=n_links, graph_hash=graph_hash)


def query_concept_fn(extractor: Extractor, linker: UMLSLinker):
    """Compose extractor + linker into a query -> {CUI} function for retrievers."""
    def fn(query: str) -> set[str]:
        return linker.concepts(extractor.extract(query))
    return fn
