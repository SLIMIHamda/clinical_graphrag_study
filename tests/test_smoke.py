import pytest

from manifest.manifest import load_manifest
from mgr.data.loader import write_items_fixture
from mgr.retrieval.factory import ConditionNotWired, build_retriever
from mgr.smoke import run_smoke


class FakeGenClient:
    def __init__(self, answer="A"):
        self.answer = answer

    def complete_text(self, model, messages, **params):
        # echo whether any context was supplied, so No-RAG vs BM25 differ in tokens
        had_ctx = any("Context:" in m["content"] for m in messages)
        return self.answer, {"in": 200 if had_ctx else 50, "out": 1}


@pytest.fixture
def env(tmp_path):
    data_root = tmp_path / "data"
    rows = [
        {"qid": f"mmlu_{i:03d}", "question": "aspirin for myocardial infarction?",
         "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": "A"}
        for i in range(25)
    ]
    write_items_fixture("MMLU-Med", data_root, rows)
    corpus = [
        {"id": "d1", "text": "aspirin and reperfusion for myocardial infarction"},
        {"id": "d2", "text": "metformin for diabetes mellitus"},
    ]
    return data_root, corpus


def test_factory_wires_smoke_conditions(env):
    _, corpus = env
    from mgr.retrieval.base import NullRetriever
    from mgr.retrieval.bm25 import BM25Retriever

    assert isinstance(build_retriever("No-RAG"), NullRetriever)
    assert isinstance(build_retriever("BM25", corpus), BM25Retriever)
    with pytest.raises(ConditionNotWired):
        build_retriever("Hybrid-CARRF", corpus)


def test_run_smoke_both_arms_complete(env, tmp_path):
    data_root, corpus = env
    m = load_manifest()
    records = run_smoke(
        m,
        FakeGenClient(answer="A"),
        data_root=data_root,
        corpus_records=corpus,
        benchmark="MMLU-Med",
        seed=42,
        n_items=25,
        results_root=tmp_path / "smoke",
    )
    assert {r.condition for r in records} == {"No-RAG", "BM25"}
    assert all(r.status == "Done" for r in records)
    assert all(r.n_items == 25 for r in records)
    # all golds "A" with constant "A" prediction -> perfect accuracy in the smoke
    assert all(r.metrics["generation"]["accuracy"] == 1.0 for r in records)

    norag = next(r for r in records if r.condition == "No-RAG")
    bm25 = next(r for r in records if r.condition == "BM25")
    # BM25 supplies context -> more input tokens than closed-book No-RAG
    assert bm25.tokens["in"] > norag.tokens["in"]
