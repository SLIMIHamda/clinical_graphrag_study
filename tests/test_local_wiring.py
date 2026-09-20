"""Runtime wiring: what `build_resources` resolves from an environment.

The regression these guard: the previous entrypoint wired 2 of 8 Resources
fields and gated dense retrieval behind NIM_API_KEY, so on a box with no key
every graph and hybrid arm raised ConditionNotWired and the sweep reported one
line of skip output while calling every row it did run Done.
"""

import json

import numpy as np
import pytest

from manifest.manifest import load_manifest
from mgr.data.loader import write_items_fixture
from mgr.graph.build import build_graph
from mgr.graph.extract import NgramExtractor
from mgr.graph.store import InMemoryGraphStore, save_snapshot
from mgr.graph.umls import UMLSLinker
from mgr.rerank.care_gate import CareGate, CareFeatures
from mgr.retrieval.factory import ConditionNotWired
from mgr.sweep import build_resources, preflight, run_sweep

VOCAB = {"heart failure": "C0018801", "metformin": "C0025598"}
CORPUS = [
    {"id": "d1", "text": "heart failure treated with metformin"},
    {"id": "d2", "text": "metformin dosing guidance"},
    {"id": "d3", "text": "unrelated administrative text"},
]


def _fitted_care_gate():
    n = len(CareFeatures.__dataclass_fields__)
    return CareGate.fit(
        [CareFeatures(*[float(i)] * n) for i in range(6)], [0, 0, 0, 1, 1, 1], epochs=50
    )


@pytest.fixture
def artifacts(tmp_path):
    """A fully built local environment: corpus, dense index, graph, UMLS dict."""
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in CORPUS), encoding="utf-8")

    emb = tmp_path / "corpus_emb.npy"
    ids = tmp_path / "corpus_ids.txt"
    np.save(emb, np.eye(len(CORPUS), 4, dtype=np.float32))
    ids.write_text("\n".join(r["id"] for r in CORPUS), encoding="utf-8")

    umls = tmp_path / "umls.json"
    umls.write_text(json.dumps(VOCAB), encoding="utf-8")

    store = InMemoryGraphStore()
    build_graph(CORPUS, store, NgramExtractor.from_linker_dict(VOCAB), UMLSLinker(VOCAB, fuzzy_enabled=False))
    snapshot = tmp_path / "graph.json"
    save_snapshot(store, snapshot)

    return {
        "DATA_ROOT": str(tmp_path),
        "CORPUS": str(corpus),
        "DENSE_EMB": str(emb),
        "DENSE_IDS": str(ids),
        "UMLS_DICT": str(umls),
        "GRAPH_SNAPSHOT": str(snapshot),
    }


# -- the headline: no NIM key, every component still wired -------------------

def test_full_local_profile_wires_every_component(artifacts):
    res, notes = build_resources(artifacts)  # note: no NIM_API_KEY at all
    assert set(res.retrievers) == {"lexical", "dense", "graph"}
    assert res.query_concepts_fn is not None
    assert res.candidate_concepts is not None
    assert res.reranker is not None
    assert res.gate is not None and res.gate.enabled
    assert "lexical" in notes and "dense" in notes and "graph" in notes


def test_every_manifest_condition_we_claim_to_support_is_buildable(artifacts, tmp_path):
    gate_path = tmp_path / "care.json"
    _fitted_care_gate().save(gate_path)

    res, _ = build_resources({**artifacts, "CARE_GATE": str(gate_path)})
    m = load_manifest()
    unwired = preflight(m, {"H2": True, "G3": True, "P3": True}, res)
    # RRF4 needs Contriever/SPECTER, which are genuinely not wired anywhere
    assert set(unwired) <= {"Hybrid-RRF4"}


def test_care_arm_is_reported_unwired_not_crashed_without_a_gate(artifacts):
    """Adaptive rerank validates in FusionRetriever.__post_init__ and raises
    ValueError. Unconverted that aborts the sweep mid-run instead of landing in
    the unwired report."""
    res, _ = build_resources(artifacts)  # no CARE_GATE
    unwired = preflight(load_manifest(), {"H2": True, "G3": True, "P3": True}, res)
    assert "Hybrid-CARRF-CARe" in unwired
    assert "CARe gate" in unwired["Hybrid-CARRF-CARe"]


def test_dense_no_longer_requires_a_nim_key(artifacts):
    """The specific bug: dense was gated on NIM_API_KEY, which must be unset for
    runs over credentialed data."""
    res, _ = build_resources(artifacts)
    assert "dense" in res.retrievers


def test_local_query_encoder_is_not_the_document_encoder(artifacts):
    from mgr.clients.local_embed import MEDCPT_QUERY

    res, _ = build_resources(artifacts)
    assert res.retrievers["dense"].embedder.model_id == MEDCPT_QUERY


def test_nim_key_still_selects_the_hosted_path(artifacts):
    """Additive, not a replacement: an existing hosted setup keeps working."""
    from mgr.clients.nim_adapters import NimEmbedder

    res, _ = build_resources({**artifacts, "NIM_API_KEY": "nvapi-test", "RERANK_BACKEND": "nim"})
    assert isinstance(res.retrievers["dense"].embedder, NimEmbedder)


def test_emb_backend_overrides_key_detection(artifacts):
    from mgr.clients.local_embed import LocalEmbedder

    res, _ = build_resources({**artifacts, "NIM_API_KEY": "nvapi-test", "EMB_BACKEND": "local"})
    assert isinstance(res.retrievers["dense"].embedder, LocalEmbedder)


# -- graceful degradation, with a reason -------------------------------------

def test_missing_artifacts_leave_components_unwired_and_say_why(tmp_path):
    res, notes = build_resources({"DATA_ROOT": str(tmp_path)})
    assert res.retrievers == {}
    assert "DENSE_EMB" in notes["dense"]
    assert "UMLS_DICT" in notes["graph"]


def test_graph_without_a_store_is_unwired_even_with_a_dictionary(tmp_path):
    umls = tmp_path / "umls.json"
    umls.write_text(json.dumps(VOCAB), encoding="utf-8")
    res, notes = build_resources({"DATA_ROOT": str(tmp_path), "UMLS_DICT": str(umls)})
    assert "graph" not in res.retrievers
    assert "GRAPH_SNAPSHOT" in notes["graph"]


def test_candidate_concepts_report_grounding_coverage(artifacts):
    _, notes = build_resources(artifacts)
    assert notes["graph"].startswith("2/3")  # d3 has no vocabulary terms


# -- gate capture ------------------------------------------------------------

def test_gate_capture_is_on_by_default(artifacts):
    res, notes = build_resources(artifacts)
    assert res.gate.enabled and res.gate.n_samples == 5
    assert "No-RAG arm only" in notes["gate_capture"]


def test_gate_capture_can_be_disabled_and_says_what_that_costs(artifacts):
    res, notes = build_resources({**artifacts, "GATE_CAPTURE": "0"})
    assert res.gate is None
    assert "Phase-A" in notes["gate_capture"]


def test_q_emb_is_wired_only_when_asked_for(artifacts):
    assert build_resources(artifacts)[0].gate.embed_client is None
    assert build_resources({**artifacts, "GATE_Q_EMB": "1"})[0].gate.embed_client is not None


def test_q_emb_shim_matches_the_embeddings_response_shape():
    """GateCapture expects the NIM `.embeddings()` envelope, not a bare array."""
    from mgr.sweep import _EmbeddingsShim

    shim = _EmbeddingsShim(lambda texts: np.ones((len(texts), 3), dtype=np.float32))
    resp = shim.embeddings("m", ["hello", "world"])
    assert len(resp["data"]) == 2
    assert resp["data"][0]["embedding"] == [1.0, 1.0, 1.0]
    assert all(isinstance(x, float) for x in resp["data"][0]["embedding"])


# -- care gate ---------------------------------------------------------------

def test_care_gate_round_trips_through_env(artifacts, tmp_path):
    gate = _fitted_care_gate()
    path = tmp_path / "care.json"
    gate.save(path)

    res, notes = build_resources({**artifacts, "CARE_GATE": str(path)})
    assert res.care_gate is not None
    assert "loaded" in notes["care_gate"]

    probe = CareFeatures(*[2.0] * len(CareFeatures.__dataclass_fields__))
    assert res.care_gate.predict_proba(probe) == pytest.approx(gate.predict_proba(probe), abs=1e-9)


# -- strict sweep ------------------------------------------------------------

class FakeGen:
    model = "fake"

    def complete_text(self, model, messages, **kw):
        return "A", {"in": 10, "out": 1}


def _tiny_manifest():
    m = load_manifest()
    m.runs = [
        r for r in m.runs
        if r.benchmark == "MMLU-Med" and r.seed == 42 and r.condition in {"No-RAG", "BM25"}
    ]
    return m


def test_strict_sweep_refuses_before_running_when_an_arm_is_unwired(tmp_path):
    m = _tiny_manifest()
    write_items_fixture("MMLU-Med", tmp_path, [
        {"qid": f"q{i}", "question": "?", "options": {"A": "a", "B": "b"}, "answer": "A"} for i in range(3)
    ])
    res, _ = build_resources({"DATA_ROOT": str(tmp_path)})  # no CORPUS -> BM25 unwired
    res.gen_client = FakeGen()
    res.gate = None
    res.n_items = 3

    with pytest.raises(ConditionNotWired, match="strict sweep"):
        run_sweep(m, {"H2": True, "G3": False, "P3": False}, res,
                  results_root=tmp_path / "results", strict=True)

    # nothing was spent: the refusal happens in pre-flight
    assert not (tmp_path / "results" / "arms").exists()


def test_non_strict_sweep_keeps_the_old_skip_behaviour(tmp_path):
    m = _tiny_manifest()
    write_items_fixture("MMLU-Med", tmp_path, [
        {"qid": f"q{i}", "question": "?", "options": {"A": "a", "B": "b"}, "answer": "A"} for i in range(3)
    ])
    res, _ = build_resources({"DATA_ROOT": str(tmp_path)})
    res.gen_client = FakeGen()
    res.gate = None
    res.n_items = 3

    records = run_sweep(m, {"H2": True, "G3": False, "P3": False}, res,
                        results_root=tmp_path / "results")
    assert {r.condition for r in records} == {"No-RAG"}
