import pytest

from mgr.retrieval.ca_rrf import (
    ca_rrf,
    concept_overlap_scores,
    concept_ranked_list,
)
from mgr.retrieval.rrf import fused_order, ranks_of, reciprocal_rank_fusion


# ----- RRF ----------------------------------------------------------------- #

def test_rrf_rewards_agreement_across_lists():
    lex = ["d1", "d2", "d3"]
    den = ["d1", "d3", "d2"]
    order = fused_order([lex, den], k=60)
    assert order[0] == "d1"  # top of both lists


def test_rrf_exact_scores():
    # single list, k=60: score = 1/(60+rank)
    fused = reciprocal_rank_fusion([["a", "b"]], k=60)
    d = dict(fused)
    assert d["a"] == pytest.approx(1 / 61)
    assert d["b"] == pytest.approx(1 / 62)


def test_rrf_weights_shift_ranking():
    lex = ["a", "b"]
    den = ["b", "a"]
    # upweighting dense flips the tie toward dense's top doc
    order = fused_order([lex, den], k=60, weights=[0.1, 1.0])
    assert order[0] == "b"


def test_rrf_deterministic_tiebreak_by_id():
    fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
    # perfect symmetry -> equal scores -> id order
    assert [d for d, _ in fused] == ["a", "b"]


def test_rrf_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


# ----- concept overlap ----------------------------------------------------- #

def test_concept_overlap_count_and_jaccard():
    q = {"mi", "aspirin"}
    cand = {"d1": {"mi"}, "d2": set(), "d3": {"mi", "aspirin", "statin"}}
    cnt = concept_overlap_scores(q, cand, metric="count")
    assert cnt == {"d1": 1.0, "d2": 0.0, "d3": 2.0}
    jac = concept_overlap_scores(q, cand, metric="jaccard")
    assert jac["d3"] == pytest.approx(2 / 3)  # |{mi,aspirin}| / |{mi,aspirin,statin}|


def test_concept_ranked_list_drops_zero_overlap():
    q = {"mi", "aspirin"}
    cand = {"d1": {"mi"}, "d2": set(), "d3": {"mi", "aspirin"}}
    assert concept_ranked_list(q, cand) == ["d3", "d1"]  # d2 excluded


# ----- CA-RRF: the isolable marginal value --------------------------------- #

def test_ca_rrf_rescues_concept_heavy_doc_vs_plain_rrf():
    # d3 is bottom-ranked lexically/densely, but is the only full concept match.
    components = {"lexical": ["d1", "d2", "d3"], "dense": ["d2", "d1", "d3"]}
    q = {"mi", "aspirin"}
    cand = {"d1": {"mi"}, "d2": set(), "d3": {"mi", "aspirin"}}

    plain = ca_rrf(components, q, cand, use_concept=False)   # ablation baseline
    aware = ca_rrf(components, q, cand, use_concept=True)    # + concept list

    r_plain = ranks_of(plain)
    r_aware = ranks_of(aware)
    # everything else frozen: the concept list is the ONLY difference
    assert r_plain["d3"] == 3                 # last without concepts
    assert r_aware["d3"] < r_plain["d3"]      # promoted with concepts
    assert r_aware["d3"] == 2


def test_ca_rrf_use_concept_false_equals_plain_rrf():
    components = {"lexical": ["a", "b", "c"], "dense": ["c", "b", "a"]}
    q = {"x"}
    cand = {"a": {"x"}, "b": set(), "c": set()}
    ablation = ca_rrf(components, q, cand, use_concept=False)
    direct = reciprocal_rank_fusion([components["lexical"], components["dense"]], k=60)
    assert ablation == direct
