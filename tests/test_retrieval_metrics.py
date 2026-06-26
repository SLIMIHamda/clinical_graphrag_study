import pytest

from mgr.metrics.retrieval import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score,
)
from mgr.stats.rgd import decompose, divergent_systems


# ----- retrieval metrics --------------------------------------------------- #

def test_recall_and_precision_at_k():
    retrieved = ["d1", "d9", "d3", "d7"]
    relevant = {"d3", "d5"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)   # d3 of {d3,d5}
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)  # ideal order
    # relevant doc sits at rank 2 only -> < 1
    val = ndcg_at_k(["x", "a", "y"], {"a"}, 3)
    assert 0.0 < val < 1.0


def test_score_aggregates_over_items():
    items = [
        (["d1", "d2", "d3"], {"d1"}),
        (["d4", "d5", "d6"], {"d6"}),
    ]
    s = score(items)
    assert s.n == 2
    assert s.recall[1] == pytest.approx(0.5)   # first query hits @1, second doesn't
    assert s.mrr == pytest.approx((1.0 + 1 / 3) / 2)


# ----- RGD decomposition (C1) ---------------------------------------------- #

def test_rgd_flags_retrieval_generation_divergence():
    # Graph-only: best retrieval, worse generation than the BM25 baseline.
    systems = {
        "BM25": (0.50, 0.60),
        "Graph-only": (0.70, 0.55),    # retrieval up, generation down -> diverges
        "Hybrid-CARRF": (0.68, 0.66),  # both up
    }
    points = decompose(systems, baseline="BM25")
    by = {p.system: p for p in points}
    assert by["Graph-only"].retrieval_gain == pytest.approx(0.20)
    assert by["Graph-only"].generation_gain == pytest.approx(-0.05)
    assert by["Graph-only"].diverges
    assert not by["Hybrid-CARRF"].diverges
    assert divergent_systems(points) == ["Graph-only"]
    # sorted by retrieval gain desc; baseline has zero gains
    assert points[0].system == "Graph-only"
    assert by["BM25"].retrieval_gain == 0.0
