import numpy as np
import pytest

from mgr.rerank.care_gate import (
    CareGate,
    cost_quality_frontier,
    extract_features,
    oracle_benefit,
)
from mgr.rerank.cross_encoder import CrossEncoderReranker, constant_scorer


# ----- features ------------------------------------------------------------ #

def test_clear_winner_has_large_gap_few_ties():
    f = extract_features([1.0, 0.2, 0.1, 0.05], tie_tol=0.05)
    assert f.top1_gap > 0.7
    assert f.near_tie_frac == pytest.approx(0.25)  # only the top doc is "near" itself


def test_ambiguous_query_has_small_gap_many_ties():
    f = extract_features([1.0, 0.99, 0.98, 0.97], tie_tol=0.05)
    assert f.top1_gap < 0.05
    assert f.near_tie_frac == 1.0  # everything within tol of the top


def test_overlap_entropy_high_when_spread():
    spread = extract_features([1, 1, 1], overlaps=[1.0, 1.0, 1.0])
    peaked = extract_features([1, 1, 1], overlaps=[1.0, 0.0, 0.0])
    assert spread.overlap_entropy > peaked.overlap_entropy
    assert peaked.overlap_entropy == 0.0


def test_oracle_benefit_label():
    assert oracle_benefit(0.8, 0.6) == 1
    assert oracle_benefit(0.6, 0.6) == 0
    assert oracle_benefit(0.5, 0.7) == 0


# ----- gate training ------------------------------------------------------- #

def _synthetic_training_set(n=400, seed=1):
    """Ambiguous queries (small gap, many ties) benefit from reranking."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for _ in range(n):
        ambiguous = rng.random() < 0.5
        if ambiguous:
            scores = [1.0, 0.99, 0.98, 0.97, 0.96]
        else:
            scores = [1.0, 0.3, 0.15, 0.1, 0.05]
        feats.append(extract_features(scores, tie_tol=0.05))
        labels.append(1 if ambiguous else 0)  # oracle: rerank helps when ambiguous
    return feats, labels


def test_gate_learns_to_fire_on_ambiguous_queries():
    feats, labels = _synthetic_training_set()
    gate = CareGate.fit(feats, labels, epochs=1500)

    ambiguous = extract_features([1.0, 0.99, 0.98, 0.97, 0.96], tie_tol=0.05)
    clear = extract_features([1.0, 0.3, 0.15, 0.1, 0.05], tie_tol=0.05)

    assert gate.predict_proba(ambiguous) > 0.5
    assert gate.predict_proba(clear) < 0.5
    assert gate.decide(ambiguous) is True
    assert gate.decide(clear) is False


def test_cost_aware_rule_suppresses_when_cost_exceeds_gain():
    feats, labels = _synthetic_training_set()
    gate = CareGate.fit(feats, labels, epochs=1500)
    ambiguous = extract_features([1.0, 0.99, 0.98, 0.97, 0.96], tie_tol=0.05)
    p = gate.predict_proba(ambiguous)
    # rerank iff expected gain (p * value) beats cost; bracket the actual p
    assert gate.decide(ambiguous, value=1.0, cost=p + 0.01) is False
    assert gate.decide(ambiguous, value=1.0, cost=p - 0.01) is True


# ----- frontier ------------------------------------------------------------ #

def test_care_matches_always_quality_at_lower_cost():
    # 4 queries: reranking helps only the first two.
    q_with = [0.9, 0.9, 0.7, 0.7]
    q_without = [0.6, 0.6, 0.7, 0.7]
    decisions = [True, True, False, False]  # a perfect gate
    fr = cost_quality_frontier(decisions, q_with, q_without, rerank_cost=1.0)

    assert fr["care"].mean_quality == pytest.approx(fr["always"].mean_quality)
    assert fr["care"].total_cost < fr["always"].total_cost      # 2 vs 4
    assert fr["care"].mean_quality > fr["never"].mean_quality
    assert fr["never"].total_cost == 0.0


# ----- cross-encoder ------------------------------------------------------- #

def test_cross_encoder_reorders_by_scorer():
    passages = {"d1": "irrelevant", "d2": "aspirin myocardial infarction", "d3": "x"}
    scorer = lambda q, p: float(len(set(q.split()) & set(p.split())))
    ce = CrossEncoderReranker(scorer=scorer)
    out = ce.rerank("aspirin myocardial infarction", ["d1", "d2", "d3"], passages)
    assert out[0] == "d2"


def test_constant_scorer_is_stable_passthrough():
    ce = CrossEncoderReranker(scorer=constant_scorer(0.0))
    assert ce.rerank("q", ["d3", "d1", "d2"], {}) == ["d1", "d2", "d3"]  # id tie-break
