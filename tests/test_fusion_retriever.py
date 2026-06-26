import pytest

from mgr.retrieval.base import RetrievalResult
from mgr.retrieval.fusion import FusionRetriever, build_fusion
from mgr.rerank.cross_encoder import CrossEncoderReranker


class FakeRetriever:
    """Returns a fixed ranked id list, ignoring the query."""

    def __init__(self, ids):
        self._ids = ids

    def retrieve(self, query, *, depth_k=10):
        return RetrievalResult(retrieved_ids=list(self._ids[:depth_k]))


class FakeGate:
    """Duck-typed CARe gate: fires when there are many near-ties."""

    def __init__(self, fire):
        self._fire = fire

    def decide(self, f, *, value=1.0, cost=0.0):
        return self._fire


PASSAGES = {f"d{i}": f"passage {i}" for i in range(6)}
CONCEPTS = {"d1": {"mi"}, "d3": {"mi", "aspirin"}, "d2": set(), "d0": set(), "d4": set(), "d5": set()}


def _components():
    return {
        "lexical": FakeRetriever(["d1", "d2", "d3"]),
        "dense": FakeRetriever(["d2", "d1", "d3"]),
    }


def test_rrf_only_no_concept():
    fr = FusionRetriever(_components(), PASSAGES, use_concept=False)
    res = fr.retrieve("q", depth_k=3)
    assert res.retrieved_ids[0] in {"d1", "d2"}
    assert not res.rerank_fired


def test_carrf_promotes_concept_match():
    no_concept = FusionRetriever(_components(), PASSAGES, use_concept=False).retrieve("q", depth_k=3)
    with_concept = FusionRetriever(
        _components(), PASSAGES, use_concept=True,
        query_concepts_fn=lambda q: {"mi", "aspirin"}, candidate_concepts=CONCEPTS,
    ).retrieve("q", depth_k=3)
    # d3 is the full concept match: it ranks no better without concepts than with
    assert with_concept.retrieved_ids.index("d3") <= no_concept.retrieved_ids.index("d3")


def test_grounding_ablation_toggles_concept():
    fr = build_fusion(
        "Hybrid-CARRF-noGrounding", components=_components(), passages=PASSAGES,
        query_concepts_fn=lambda q: {"mi"}, candidate_concepts=CONCEPTS,
    )
    assert fr.use_concept is False  # noGrounding turns the concept list off


def test_static_rerank_always_fires():
    # reranker reverses by passage-id length tiebreak; just assert it fires
    fr = build_fusion(
        "Hybrid-CARRF-staticRerank", components=_components(), passages=PASSAGES,
        query_concepts_fn=lambda q: {"mi"}, candidate_concepts=CONCEPTS,
        reranker=CrossEncoderReranker(scorer=lambda q, p: 0.0),
    )
    res = fr.retrieve("q", depth_k=3)
    assert res.rerank_fired is True


def test_adaptive_rerank_respects_gate():
    comps = _components()
    common = dict(
        components=comps, passages=PASSAGES, query_concepts_fn=lambda q: {"mi"},
        candidate_concepts=CONCEPTS, reranker=CrossEncoderReranker(scorer=lambda q, p: 0.0),
    )
    fired = build_fusion("Hybrid-CARRF-CARe", care_gate=FakeGate(True), **common).retrieve("q")
    held = build_fusion("Hybrid-CARRF-CARe", care_gate=FakeGate(False), **common).retrieve("q")
    assert fired.rerank_fired is True
    assert held.rerank_fired is False


def test_adaptive_requires_gate():
    with pytest.raises(ValueError):
        build_fusion(
            "Hybrid-CARRF-CARe", components=_components(), passages=PASSAGES,
            query_concepts_fn=lambda q: {"mi"}, candidate_concepts=CONCEPTS,
            reranker=CrossEncoderReranker(scorer=lambda q, p: 0.0),
        )


def test_build_fusion_rejects_non_hybrid():
    with pytest.raises(KeyError):
        build_fusion("No-RAG", components=_components(), passages=PASSAGES)
