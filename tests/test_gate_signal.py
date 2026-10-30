import json
import math

from mgr.gate.features import (
    compute_features,
    confidence_features,
    label_variants,
    option_distribution,
    option_logprobs,
    option_probs_from_logprobs,
    self_consistency_features,
    structural_features,
)
from mgr.analysis.gate_signal import gate_signal_analysis, load_records


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
    assert abs(f["entropy"] - 0.8018 / math.log(4)) < 1e-3   # normalized by log|O|


def test_entropy_is_on_one_scale_across_option_counts():
    for n in (2, 3, 4, 5):
        f = confidence_features({str(i): 1.0 / n for i in range(n)})
        assert abs(f["entropy"] - 1.0) < 1e-9   # uniform -> 1 whatever |O| is
    assert confidence_features({"A": 1.0})["entropy"] == 0.0


def test_label_variants_per_answer_type():
    assert label_variants("mcq", ["A", "B", "C", "D", "E"]) == {c: (c,) for c in "ABCDE"}
    assert label_variants("mcq") == {c: (c,) for c in "ABCD"}
    ynm = label_variants("yes_no_maybe")
    assert set(ynm) == {"yes", "no", "maybe"} and ynm["yes"] == ("yes", "Yes")
    assert set(label_variants("yes_no")) == {"yes", "no"}
    assert label_variants("free") == {}


def test_option_distribution_normalizes_and_reports_valid_mass():
    lps = {"A": math.log(0.5), "B": math.log(0.2), "C": math.log(0.1), "D": math.log(0.1)}
    probs, valid = option_distribution(lps)
    assert abs(valid - 0.9) < 1e-9                       # 10% of the mass went off-format
    assert abs(probs["A"] - 0.5 / 0.9) < 1e-9            # conditioned on a valid label
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert option_distribution({}) == ({}, 0.0)


def test_option_logprobs_folds_surface_forms():
    scores = {"yes": math.log(0.3), "Yes": math.log(0.3), "no": math.log(0.2), "No": math.log(0.1)}
    lp = option_logprobs(scores, {"yes": ("yes", "Yes"), "no": ("no", "No")})
    assert abs(math.exp(lp["yes"]) - 0.6) < 1e-9 and abs(math.exp(lp["no"]) - 0.3) < 1e-9


def test_self_consistency_counts_unparsed_samples():
    f = self_consistency_features(["A", None, "", "A"], greedy_answer="A")
    assert abs(f["sc_agreement"] - 0.5) < 1e-9           # A:2 vs unparsed:2, not A:2 of 2
    assert abs(f["sc_matches_greedy"] - 0.5) < 1e-9
    assert abs(f["sc_entropy"] - math.log(2)) < 1e-9


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
    # the net-benefit gate ran too, but 10 breaks is flagged as too few
    assert "accuracy" in out["learned_gates"]["net_benefit"]
    assert out["underpowered_breaks"] is True
    assert any("UNDERPOWERED (breaks)" in n for n in out["notes"])


def test_net_benefit_gate_skipped_without_enough_breaks():
    recs = [_rec(1, 1, 0.1), _rec(1, 0, 0.1), _rec(0, 1, 0.9), _rec(0, 0, 0.1)]
    learned = gate_signal_analysis(recs)["learned_gates"]
    assert "skipped" in learned["net_benefit"]["note"]
    assert "skipped" in learned["net_benefit_tuned"]["note"]


def test_net_benefit_gate_uses_breaks_the_rescue_classifier_cannot_see():
    # Low entropy: rescue 1/4, both wrong 1/4, both right 1/2 -> P(R)-P(B) = +0.25.
    # High entropy: rescue 2/5, break 3/5                     -> P(R)-P(B) = -0.20.
    # Rescues are *more* common in the high group, so a rescue-only ranking puts
    # it first and no threshold on that ranking can retrieve the low group alone.
    recs = []
    for j in range(120):
        no_rag, rag = [(0, 1), (0, 0), (1, 1), (1, 1)][j % 4]
        recs.append(_rec(no_rag, rag, 0.2 + j * 1e-4))
    for j in range(120):
        no_rag, rag = [(0, 1), (1, 0), (0, 1), (1, 0), (1, 0)][j % 5]
        recs.append(_rec(no_rag, rag, 0.8 + j * 1e-4))
    out = gate_signal_analysis(recs, lam=0.0, seed=0, n_seeds=3, use_embedding=False)

    assert out["n_rescueable"] == 78 and out["n_breaks"] == 72
    assert out["underpowered"] is False and out["underpowered_breaks"] is False
    b = out["baseline"]
    assert b["no_rag_acc"] == 0.55 and b["rag_always_acc"] == 0.575 and b["oracle_acc"] == 0.875
    nb = out["learned_gates"]["net_benefit"]
    # plug-in rule retrieves the low-entropy group only: (90 + 72) / 240 correct
    assert abs(nb["accuracy"] - 0.675) < 0.02
    assert abs(nb["retrieval_rate"] - 0.5) < 0.05
    assert abs(nb["recovered_fraction"] - 0.125 / 0.325) < 0.06
    assert nb["oof_auroc_break"] > 0.75
    # best a rescue-only ranking can do here is retrieve everything (0.575)
    assert out["learned_gates"]["logistic"]["accuracy"] <= 0.6
    assert nb["accuracy"] > out["learned_gates"]["logistic"]["accuracy"] + 0.05
    assert out["learned_gates"]["net_benefit_tuned"]["accuracy"] > 0.62


def test_load_records_skips_failed_probes(tmp_path):
    p = tmp_path / "features.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"qid": "a", "no_rag_correct": 1, "rag_correct": 0, "entropy": 0.1},
        {"qid": "b", "em": 0, "rag_correct": 1, "gate_error": "ValueError: boom"},
    ]), encoding="utf-8")
    assert [r["qid"] for r in load_records(p)] == ["a"]


def test_empty_records_returns_empty():
    assert gate_signal_analysis([]) == {}
    assert gate_signal_analysis([{"entropy": 0.5}]) == {}  # missing outcomes -> dropped
