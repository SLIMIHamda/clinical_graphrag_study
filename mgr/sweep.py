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
from mgr.generate.executor import GateCapture, GenClient, RAGExecutor
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
    gate: GateCapture | None = None  # Gate-A feature capture; only the No-RAG arm acts on it


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
        try:
            return build_fusion(
                condition, components=comps, passages=res.passages,
                query_concepts_fn=res.query_concepts_fn, candidate_concepts=res.candidate_concepts,
                care_gate=res.care_gate, reranker=res.reranker, k=res.k, weights=res.weights,
            )
        except ValueError as e:
            # FusionRetriever validates its own prerequisites (a CARe gate for
            # adaptive rerank, a concept fn for CA-RRF). Those are missing
            # *resources*, so they belong in the unwired report the sweep already
            # keeps — raised raw they would abort mid-sweep instead.
            raise ConditionNotWired(f"{condition}: {e}") from e

    raise ConditionNotWired(f"no arm wired for condition {condition!r}")


def build_arm(condition: str, res: Resources) -> RAGExecutor:
    return RAGExecutor(
        client=res.gen_client,
        data_root=res.data_root,
        retriever=build_retriever_for(condition, res),
        n_items=res.n_items,
        gate=res.gate,  # executor self-gates: only the No-RAG (NullRetriever) arm captures
    )


def preflight(
    manifest: Manifest, gate_ledger: dict[str, bool], resources: Resources
) -> dict[str, str]:
    """Condition -> why it is unwired, for every distinct Ready condition.

    Cheap (it builds one arm per *condition*, not per row) and pure, so it can be
    run before anything is spent.
    """
    unwired: dict[str, str] = {}
    for condition in sorted({row.condition for row in manifest.resolve_ready(gate_ledger)}):
        try:
            build_arm(condition, resources)
        except ConditionNotWired as e:
            unwired[condition] = str(e)
    return unwired


def run_sweep(
    manifest: Manifest,
    gate_ledger: dict[str, bool],
    resources: Resources,
    *,
    results_root: str | Path = "results",
    manifest_lock_hash: str | None = None,
    strict: bool = False,
) -> list[rec.RunRecord]:
    """Run every Ready row whose condition is wired; skip+log the rest.

    ``strict`` turns an unwired condition into a pre-flight failure instead of a
    one-line note at the end. A sweep that quietly drops half its arms still
    reports every row it *did* run as Done, which reads as a clean result — so a
    long run you intend to be complete should set this.
    """
    if strict:
        unwired = preflight(manifest, gate_ledger, resources)
        if unwired:
            detail = "; ".join(f"{c}: {why}" for c, why in unwired.items())
            raise ConditionNotWired(f"strict sweep has {len(unwired)} unwired condition(s) -> {detail}")

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


# --------------------------------------------------------------------------- #
# Runtime entrypoint: assemble Resources from an environment mapping and sweep.
#
# Every component is *capability-detected*, never mode-switched: each one takes
# the local implementation or the hosted one depending on what the environment
# actually offers. There is no "MIMIC mode" — a box with a NIM key keeps the
# hosted path, a box without one gets the in-process path, and the same code
# serves both. That matters because the PhysioNet DUA forbids sending chart text
# to a hosted endpoint, so on the clinical runs `NIM_API_KEY` is deliberately
# absent, and anything that silently required it would simply not be wired.
#
# `build_resources` takes the env as an argument so this is testable; the old
# entrypoint read os.environ directly and was marked "not unit-tested", which is
# how it came to wire only 2 of the 8 Resources fields — every graph and hybrid
# arm raised ConditionNotWired and was counted in a single line of skip output.
# --------------------------------------------------------------------------- #

WiringNotes = dict[str, str]


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(env: Mapping[str, str], key: str) -> Path | None:
    raw = env.get(key)
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def _build_gen_client(env: Mapping[str, str]) -> GenClient:
    """vLLM over HTTP (default) or an in-process transformers model.

    Both expose ``complete_text_logprobs``, which Gate-A capture requires.
    """
    backend = env.get("GEN_BACKEND", "vllm").strip().lower()
    if backend in {"hf_local", "local", "hf"}:
        from mgr.clients.local_hf import LocalHFClient

        return LocalHFClient(
            model_id=env.get("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
            max_new_tokens=int(env.get("MAX_NEW_TOKENS", "8")),
        )
    from mgr.clients.vllm import VLLMClient

    return VLLMClient(
        base_url=env.get("VLLM_BASE_URL", "http://localhost:8000"),
        api_key=env.get("VLLM_API_KEY"),
    )


def _build_embedder(env: Mapping[str, str], *, for_queries: bool = True):
    """The query-side embedder: local encoder, or NIM when a key is present.

    ``EMB_BACKEND`` forces the choice; otherwise a NIM key selects NIM and its
    absence selects local. MedCPT is asymmetric, so the query encoder is *not*
    the checkpoint the index was built with — see mgr.clients.local_embed.
    """
    backend = env.get("EMB_BACKEND", "").strip().lower()
    nim_key = env.get("NIM_API_KEY")
    if backend == "nim" or (not backend and nim_key):
        if not nim_key:
            raise ValueError("EMB_BACKEND=nim but NIM_API_KEY is unset")
        from mgr.clients.nim import NimClient
        from mgr.clients.nim_adapters import NimEmbedder

        client = NimClient(base_url=env.get("NIM_BASE_URL", "https://integrate.api.nvidia.com"), api_key=nim_key)
        return NimEmbedder(client, model=env.get("EMB_MODEL", "nvidia/nv-embedqa-e5-v5"))

    from mgr.clients.local_embed import MEDCPT_ARTICLE, MEDCPT_QUERY, LocalEmbedder

    default_model = MEDCPT_QUERY if for_queries else MEDCPT_ARTICLE
    key = "EMB_QUERY_MODEL" if for_queries else "EMB_DOC_MODEL"
    return LocalEmbedder(
        model_id=env.get(key) or env.get("EMB_MODEL") or default_model,
        batch_size=int(env.get("EMB_BATCH_SIZE", "64")),
    )


class _EmbeddingsShim:
    """Adapt an ``Embedder`` callable to the ``.embeddings()`` response shape
    :class:`~mgr.generate.executor.GateCapture` expects for the ``q_emb`` feature."""

    def __init__(self, embedder):
        self._embedder = embedder

    def embeddings(self, model: str, inputs: list[str], **_params) -> dict:
        vectors = self._embedder(list(inputs))
        return {"data": [{"embedding": [float(x) for x in row]} for row in vectors]}


def _wire_lexical(env, retrievers, passages, notes: WiringNotes) -> None:
    corpus = _path(env, "CORPUS")
    if corpus is None:
        notes["lexical"] = "CORPUS unset or missing"
        return
    from mgr.retrieval.bm25 import BM25Index, BM25Retriever
    from mgr.smoke import load_corpus

    records = load_corpus(corpus)
    retrievers["lexical"] = BM25Retriever(BM25Index.from_records(records))
    passages.update({str(r["id"]): str(r["text"]) for r in records})
    notes["lexical"] = f"BM25 over {len(records)} chunks"


def _wire_dense(env, retrievers, passages, notes: WiringNotes):
    """Precomputed matrix + a query embedder. No NIM key required."""
    emb_path, ids_path = _path(env, "DENSE_EMB"), _path(env, "DENSE_IDS")
    if emb_path is None or ids_path is None:
        notes["dense"] = "DENSE_EMB/DENSE_IDS unset or missing (run scripts/build_index.py)"
        return None
    import numpy as np

    from mgr.retrieval.dense import DenseIndex, DenseRetriever

    embedder = _build_embedder(env, for_queries=True)
    ids_list = ids_path.read_text(encoding="utf-8").split()
    index = DenseIndex.from_embeddings(ids_list, np.load(emb_path))
    retrievers["dense"] = DenseRetriever(index, embedder, passages)
    notes["dense"] = f"{len(ids_list)} vectors, dim {index.dim}, query encoder {getattr(embedder, 'model_id', 'nim')}"
    return embedder


def _wire_graph(env, retrievers, passages, notes: WiringNotes):
    """Graph store + the query -> {CUI} function both graph and CA-RRF need.

    Concept grounding at query time uses a dictionary scan, not an LLM call:
    offline, free, and permitted on chart text.
    """
    from mgr.graph.extract import NgramExtractor
    from mgr.graph.umls import UMLSLinker

    umls_path = _path(env, "UMLS_DICT")
    if umls_path is None:
        notes["graph"] = "UMLS_DICT unset or missing (term -> CUI json)"
        return None, None

    import json

    exact_map = json.loads(umls_path.read_text(encoding="utf-8"))
    linker = UMLSLinker(exact_map, fuzzy_enabled=_flag(env, "UMLS_FUZZY", False))
    extractor = NgramExtractor.from_linker_dict(exact_map)

    store = None
    snapshot = _path(env, "GRAPH_SNAPSHOT")
    if snapshot is not None:
        from mgr.graph.store import load_snapshot

        store = load_snapshot(snapshot)
    elif env.get("NEO4J_URI"):
        from mgr.graph.neo4j_store import Neo4jStore

        store = Neo4jStore(
            uri=env["NEO4J_URI"], user=env.get("NEO4J_USER", "neo4j"),
            password=env.get("NEO4J_PASSWORD", "neo4j"),
        )
    if store is None:
        notes["graph"] = "no GRAPH_SNAPSHOT and no NEO4J_URI (run scripts/build_graph.sh)"
        return None, None

    from mgr.graph.build import query_concept_fn
    from mgr.retrieval.graph_retriever import GraphRetriever

    fn = query_concept_fn(extractor, linker)
    retrievers["graph"] = GraphRetriever(store, fn, hops=int(env.get("GRAPH_HOPS", "1")))

    # CA-RRF scores candidates by concept overlap, so it needs each passage's
    # concepts. An empty map here is exactly the failure that made CA-RRF
    # identical to plain RRF in the first POC, so it is reported, not assumed.
    candidate_concepts = {pid: store.concepts_of(pid) for pid in passages}
    n_grounded = sum(1 for v in candidate_concepts.values() if v)
    notes["graph"] = f"{n_grounded}/{len(passages) or 0} passages carry concepts"
    return fn, candidate_concepts


def _wire_reranker(env, notes: WiringNotes):
    backend = env.get("RERANK_BACKEND", "local").strip().lower()
    if backend in {"none", "off"}:
        notes["reranker"] = "disabled"
        return None
    if backend == "nim":
        from mgr.clients.nim import NimClient
        from mgr.clients.nim_adapters import NimReranker

        notes["reranker"] = "NIM"
        return NimReranker(NimClient(base_url=env.get("NIM_BASE_URL", ""), api_key=env.get("NIM_API_KEY")))
    from mgr.rerank.cross_encoder import HFCrossEncoderReranker

    model_id = env.get("RERANK_MODEL", "ncbi/MedCPT-Cross-Encoder")
    notes["reranker"] = f"local {model_id}"
    return HFCrossEncoderReranker(model_id=model_id)


def _wire_care_gate(env, notes: WiringNotes):
    gate_path = _path(env, "CARE_GATE")
    if gate_path is None:
        notes["care_gate"] = "CARE_GATE unset -> Hybrid-CARRF-CARe is UNWIRED (no default gate)"
        return None
    from mgr.rerank.care_gate import CareGate

    notes["care_gate"] = f"loaded {gate_path.name}"
    return CareGate.load(gate_path)


def build_resources(env: Mapping[str, str]) -> tuple["Resources", WiringNotes]:
    """Assemble :class:`Resources` from an environment mapping.

    Returns the resources and a note per component saying what was wired or why
    it was not, so the summary the runner prints is derived from what happened
    rather than from what was intended.
    """
    from mgr.generate.executor import GateCapture

    notes: WiringNotes = {}
    retrievers: dict[str, Retriever] = {}
    passages: dict[str, str] = {}

    gen = _build_gen_client(env)
    target = getattr(gen, "model", None) or getattr(gen, "base_url", "?")
    notes["generation"] = f"{env.get('GEN_BACKEND', 'vllm')} -> {target}"

    _wire_lexical(env, retrievers, passages, notes)
    query_embedder = _wire_dense(env, retrievers, passages, notes)
    query_concepts_fn, candidate_concepts = _wire_graph(env, retrievers, passages, notes)

    gate = None
    if _flag(env, "GATE_CAPTURE", True):
        n_samples = int(env.get("GATE_N_SAMPLES", "5"))
        embed_client = None
        if query_embedder is not None and _flag(env, "GATE_Q_EMB", False):
            embed_client = _EmbeddingsShim(query_embedder)
        gate = GateCapture(
            enabled=True,
            n_samples=n_samples,
            embed_client=embed_client,
            embed_model=env.get("EMB_MODEL", "") if embed_client else "",
        )
        notes["gate_capture"] = (
            f"on, {n_samples} self-consistency draws"
            + (" + q_emb" if embed_client else "")
            + " (No-RAG arm only)"
        )
    else:
        notes["gate_capture"] = "off -- Phase-A analysis will have no features"

    resources = Resources(
        gen_client=gen,
        data_root=env.get("DATA_ROOT", "data"),
        passages=passages,
        retrievers=retrievers,
        query_concepts_fn=query_concepts_fn,
        candidate_concepts=candidate_concepts,
        care_gate=_wire_care_gate(env, notes),
        reranker=_wire_reranker(env, notes),
        k=int(env.get("RRF_K", "60")),
        n_items=(int(env["N_ITEMS"]) if env.get("N_ITEMS") else None),
        gate=gate,
    )
    return resources, notes


def build_resources_from_env() -> "Resources":
    import os

    resources, notes = build_resources(os.environ)
    for key, note in notes.items():
        print(f"[wiring] {key:<14} {note}")
    return resources


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from manifest.manifest import load_gate_ledger, load_manifest

    ap = argparse.ArgumentParser(description="Run the sweep over Ready rows")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--strict", action="store_true",
                    help="fail before running if any Ready condition is unwired")
    ap.add_argument("--preflight-only", action="store_true",
                    help="report what is wired and exit without running anything")
    args = ap.parse_args(argv)

    m = load_manifest()
    ledger = load_gate_ledger()
    resources, notes = build_resources(os.environ)
    for key, note in notes.items():
        print(f"[wiring] {key:<14} {note}")

    unwired = preflight(m, ledger, resources)
    for condition, why in unwired.items():
        print(f"[preflight] UNWIRED {condition}: {why}")
    if args.preflight_only:
        print(f"[preflight] {len(unwired)} unwired condition(s)")
        return 1 if unwired else 0

    records = run_sweep(m, ledger, resources, results_root=args.results_root, strict=args.strict)
    done = sum(1 for r in records if r.status == "Done")
    print(f"[sweep] {done}/{len(records)} runs Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
