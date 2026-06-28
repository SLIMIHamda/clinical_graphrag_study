import pytest

from manifest.manifest import load_manifest
from mgr.data.loader import write_items_fixture
from mgr.retrieval.base import NullRetriever, RetrievalResult
from mgr.retrieval.bm25 import BM25Index, BM25Retriever
from mgr.retrieval.factory import ConditionNotWired
from mgr.retrieval.fusion import FusionRetriever
from mgr.sweep import Resources, build_retriever_for, run_sweep


class FakeGen:
    def complete_text(self, model, messages, **kw):
        return "A", {"in": 10, "out": 1}


class FakeRetriever:
    def __init__(self, ids):
        self._ids = ids

    def retrieve(self, query, *, depth_k=10):
        return RetrievalResult(retrieved_ids=self._ids[:depth_k])


def _resources(tmp_path, **over):
    base = dict(
        gen_client=FakeGen(),
        data_root=tmp_path,
        passages={"d1": "t1", "d2": "t2"},
        retrievers={
            "lexical": FakeRetriever(["d1", "d2"]),
            "dense": FakeRetriever(["d2", "d1"]),
            "graph": FakeRetriever(["d1"]),
        },
        query_concepts_fn=lambda q: {"c1"},
        candidate_concepts={"d1": {"c1"}, "d2": set()},
    )
    base.update(over)
    return Resources(**base)


def test_build_norag_and_single_components(tmp_path):
    res = _resources(tmp_path)
    assert isinstance(build_retriever_for("No-RAG", res), NullRetriever)
    assert build_retriever_for("BM25", res) is res.retrievers["lexical"]
    assert build_retriever_for("Dense-MedCPT", res) is res.retrievers["dense"]


def test_build_hybrid_is_fusion(tmp_path):
    res = _resources(tmp_path)
    r = build_retriever_for("Hybrid-CARRF", res)
    assert isinstance(r, FusionRetriever)
    assert r.use_concept is True


def test_rrf4_unwired_without_contriever_specter(tmp_path):
    res = _resources(tmp_path)
    with pytest.raises(ConditionNotWired):
        build_retriever_for("Hybrid-RRF4", res)


def test_run_sweep_runs_wired_ready_rows(tmp_path):
    # tiny manifest slice: No-RAG + BM25 rows for MMLU-Med, seed 42
    m = load_manifest()
    subset = [
        r for r in m.runs
        if r.benchmark == "MMLU-Med" and r.seed == 42 and r.condition in {"No-RAG", "BM25"}
    ]
    m.runs = subset
    write_items_fixture("MMLU-Med", tmp_path, [
        {"qid": f"q{i}", "question": "?", "options": {"A": "a", "B": "b"}, "answer": "A"} for i in range(5)
    ])
    res = _resources(tmp_path, n_items=5)
    records = run_sweep(
        m, {"H2": True, "G3": False, "P3": False}, res, results_root=tmp_path / "results"
    )
    assert {r.condition for r in records} == {"No-RAG", "BM25"}
    assert all(r.status == "Done" for r in records)
