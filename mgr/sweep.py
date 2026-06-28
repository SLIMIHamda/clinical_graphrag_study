"""Sweep assembly — map each condition to its executor and run the Ready rows.

This is the integration layer: given the runtime resources (generation client,
data, component retrievers, grounding fns, CARe gate, reranker), ``build_arm``
constructs the right :class:`RAGExecutor` for a condition, and ``run_sweep``
drives every Ready row through the runner (claim → execute → record → rollup).

Conditions that need components you haven't wired (e.g. RRF4 without Contriever/
SPECTER) raise ``ConditionNotWired`` and are skipped honestly — the harness never
silently substitutes a different system for a named one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from manifest.manifest import Manifest
from mgr.generate.executor import GenClient, RAGExecutor
from mgr.rerank.care_gate import CareGate
from mgr.retrieval.base import NullRetriever, Retriever
from mgr.retrieval.factory import ConditionNotWired
from mgr.retrieval.fusion import HYBRID_SPECS, FusionRetriever, build_fusion
from mgr.runner import Runner
from mgr.tracking import record as rec

# Single-retriever conditions: which component key serves each.
SINGLE_COMPONENT = {
    "No-RAG": None,            # closed-book
    "BM25": "lexical",
    "Dense-MedCPT": "dense",
    "Graph-only": "graph",
}

# Component keys each hybrid arm fuses.
HYBRID_COMPONENTS = {
    "Hybrid-RRF2": ["lexical", "dense"],
    "Hybrid-RRF4": ["lexical", "contriever", "specter", "dense"],
    "Hybrid-CARRF": ["lexical", "dense", "graph"],
    "Hybrid-CARRF-staticRerank": ["lexical", "dense", "graph"],
    "Hybrid-CARRF-CARe": ["lexical", "dense", "graph"],
    "Hybrid-CARRF-noVecIndex": ["lexical", "dense", "graph"],   # caller injects a flat dense
    "Hybrid-CARRF-noGrounding": ["lexical", "dense", "graph"],
    "MedGraphRAG-repro": ["graph", "dense"],                    # best-effort SOTA anchor
}


@dataclass
class Resources:
    gen_client: GenClient
    data_root: str | Path
    passages: dict[str, str] = field(default_factory=dict)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    query_concepts_fn: Callable[[str], set[str]] | None = None
    candidate_concepts: Mapping[str, set[str]] | None = None
    care_gate: CareGate | None = None
    reranker: object | None = None
    k: int = 60
    weights: dict[str, float] | None = None
    n_items: int | None = None  # smoke override; None = full benchmark


def _pull(condition: str, needs: list[str], res: Resources) -> dict[str, Retriever]:
    missing = [n for n in needs if n not in res.retrievers]
    if missing:
        raise ConditionNotWired(f"{condition} needs component retrievers {missing}")
    return {n: res.retrievers[n] for n in needs}


def build_retriever_for(condition: str, res: Resources) -> Retriever:
    if condition in SINGLE_COMPONENT:
        key = SINGLE_COMPONENT[condition]
        if key is None:
            return NullRetriever()
        if key not in res.retrievers:
            raise ConditionNotWired(f"{condition} needs the {key!r} retriever")
        return res.retrievers[key]

    if condition == "MedGraphRAG-repro":
        comps = _pull(condition, HYBRID_COMPONENTS[condition], res)
        return FusionRetriever(comps, res.passages, use_concept=False, k=res.k, weights=res.weights)

    if condition in HYBRID_SPECS:
        comps = _pull(condition, HYBRID_COMPONENTS[condition], res)
        return build_fusion(
            condition, components=comps, passages=res.passages,
            query_concepts_fn=res.query_concepts_fn, candidate_concepts=res.candidate_concepts,
            care_gate=res.care_gate, reranker=res.reranker, k=res.k, weights=res.weights,
        )

    raise ConditionNotWired(f"no arm wired for condition {condition!r}")


def build_arm(condition: str, res: Resources) -> RAGExecutor:
    return RAGExecutor(
        client=res.gen_client,
        data_root=res.data_root,
        retriever=build_retriever_for(condition, res),
        n_items=res.n_items,
    )


def run_sweep(
    manifest: Manifest,
    gate_ledger: dict[str, bool],
    resources: Resources,
    *,
    results_root: str | Path = "results",
    manifest_lock_hash: str | None = None,
) -> list[rec.RunRecord]:
    """Run every Ready row whose condition is wired; skip+log the rest."""
    runner = Runner(manifest, gate_ledger, results_root, manifest_lock_hash=manifest_lock_hash)
    out: list[rec.RunRecord] = []
    skipped: dict[str, int] = {}
    for row in manifest.resolve_ready(gate_ledger):
        try:
            arm = build_arm(row.condition, resources)
        except ConditionNotWired:
            skipped[row.condition] = skipped.get(row.condition, 0) + 1
            continue
        record = runner.run_one(row.run_id, executor=arm)
        if record is not None:
            out.append(record)
    runner.rollup()
    if skipped:
        print(f"[sweep] skipped unwired conditions: {skipped}")
    return out
