import math

from mgr.gate.features import (
    compute_features,
    confidence_features,
    option_probs_from_logprobs,
    self_consistency_features,
    structural_features,
)
from mgr.analysis.gate_signal import gate_signal_analysis


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_option_probs_from_logprobs_matches_and_floors():
    top = [
        {"token": " B", "logprob": math.log(0.7)},
        {"token": "A", "logprob": math.log(0.2)},
        {"token": "C", "logprob": math.log(0.1)},
    ]
    p = option_probs_from_logprobs(top, ["A", "B", "C", "D"])
    assert max(p, key=p.get) == "B"
    assert abs(p["B"] - 0.7) < 1e-3          # renormalized, D floored ~0
    assert p["D"] > 0.0 and p["D"] < 1e-4    # unseen option kept on floor


def test_confidence_features_values():
    f = confidence_features({"A": 0.2, "B": 0.7, "C": 0.1, "D": 0.0})
    assert abs(f["confidence"] - 0.7) < 1e-9
    assert abs(f["margin"] - 0.5) < 1e-9
    assert abs(f["entropy"] - 0.8018) < 1e-3


def test_self_consistency_features():
    f = self_consistency_features(["A", "A", "A", "B"], greedy_answer="A")
    assert abs(f["sc_agreement"] - 0.75) < 1e-9
    assert abs(f["sc_matches_greedy"] - 0.75) < 1e-9
    assert f["sc_entropy"] > 0.0
    # no samples -> all zero (self-consistency disabled)
    assert self_consistency_features([], "A") == {
        "sc_agreement": 0.0, "sc_entropy": 0.0, "sc_matches_greedy": 0.0
    }


def test_compute_features_carries_embedding_and_structure():
    f = compute_features(
        option_probs={"A": 0.6, "B": 0.4},
        greedy_answer="A",
        question="one two three",
        n_options=4,
        samples=["A", "A", "B"],
        q_emb=[0.1, 0.2, 0.3],
    )
    assert f["q_len_words"] == 3.0 and f["n_options"] == 4.0
    assert f["q_emb"] == [0.1, 0.2, 0.3]
    assert "confidence" in f and "sc_agreement" in f


# --------------------------------------------------------------------------- #
# gate-signal analysis
# --------------------------------------------------------------------------- #
def _rec(no_rag, rag, entropy):
    # minimal record: one informative feature (entropy) + the two outcomes
    return {"no_rag_correct": no_rag, "rag_correct": rag, "entropy": entropy,
            "confidence": 1.0 - entropy, "margin": 1.0 - entropy, "sc_entropy": entropy}


def test_oracle_and_perfect_threshold_recovery():
    # 4 canonical items: neutral, break, rescue, miss
    recs = [_rec(1, 1, 0.1), _rec(1, 0, 0.1), _rec(0, 1, 0.9), _rec(0, 0, 0.1)]
    out = gate_signal_analysis(recs, min_pos=30)
    b = out["baseline"]
    assert b["no_rag_acc"] == 0.5 and b["rag_always_acc"] == 0.5
    assert b["oracle_acc"] == 0.75 and b["oracle_gain"] == 0.25
    assert out["n_rescueable"] == 1 and out["n_breaks"] == 1
    # entropy>tau retrieves exactly the one rescueable item -> hits the oracle
    ent = out["threshold_policies"]["entropy"]
    assert ent["accuracy"] == 0.75 and ent["recovered_fraction"] == 1.0
    # anchors: never == No-RAG, always == RAG-always, neither recovers anything
    assert out["threshold_policies"]["never_retrieve"]["recovered_fraction"] == 0.0
    assert out["threshold_policies"]["always_retrieve"]["recovered_fraction"] == 0.0
    assert out["underpowered"] is True  # 1 positive < 30


def test_signal_auroc_and_learned_gate_runs():
    # 40 deterministic items; entropy cleanly separates the rescueable class
    recs = []
    for i in range(10):
        recs.append(_rec(0, 1, 0.80 + i * 1e-3))  # rescue: high entropy
    for i in range(10):
        recs.append(_rec(1, 1, 0.20 + i * 1e-3))  # neutral: low entropy
    for i in range(10):
        recs.append(_rec(1, 0, 0.20 + i * 1e-3))  # break: low entropy
    for i in range(10):
        recs.append(_rec(0, 0, 0.50 + i * 1e-3))  # miss: mid entropy
    out = gate_signal_analysis(recs, min_pos=30, seed=0)

    assert out["n_items"] == 40 and out["n_rescueable"] == 10
    # entropy should top the univariate ranking with AUROC well above chance
    top = out["univariate_auroc"][0]
    assert top["feature"] in {"entropy", "sc_entropy", "confidence", "margin"}
    assert top["auroc"] is not None and top["auroc"] > 0.8
    # a threshold gate recovers the full oracle here (clean separation)
    assert out["threshold_policies"]["entropy"]["recovered_fraction"] == 1.0
    # learned gates actually ran (enough positives to stratify)
    assert "accuracy" in out["learned_gates"]["logistic"]
    assert out["learned_gates"]["logistic"]["cv_folds"] >= 2


def test_empty_records_returns_empty():
    assert gate_signal_analysis([]) == {}
    assert gate_signal_analysis([{"entropy": 0.5}]) == {}  # missing outcomes -> dropped
