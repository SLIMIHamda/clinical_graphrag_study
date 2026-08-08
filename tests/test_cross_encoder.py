from mgr.rerank.cross_encoder import (
    CrossEncoderReranker,
    HFCrossEncoderReranker,
    constant_scorer,
)


def test_constant_scorer_keeps_order_by_id_tiebreak():
    r = CrossEncoderReranker(constant_scorer(0.0))
    assert r.rerank("q", ["b", "a", "c"], {"a": "x", "b": "y", "c": "z"}) == ["a", "b", "c"]


def test_hf_reranker_reorders_by_injected_scores():
    """With score_batch injected, no torch is needed — this tests the reorder,
    batching call, and tie-break, which is all the notebook relies on."""
    seen = {}

    def fake_score(pairs):
        seen["pairs"] = pairs
        # score = passage length; longer passage ranks higher
        return [float(len(p[1])) for p in pairs]

    r = HFCrossEncoderReranker(score_batch=fake_score)
    passages = {"a": "short", "b": "much much longer passage", "c": "mid length"}
    out = r.rerank("query", ["a", "b", "c"], passages)

    assert out == ["b", "c", "a"]                       # by descending length
    assert seen["pairs"] == [("query", passages[c]) for c in ["a", "b", "c"]]  # query paired w/ each


def test_hf_reranker_ties_break_by_id():
    r = HFCrossEncoderReranker(score_batch=lambda pairs: [1.0] * len(pairs))
    assert r.rerank("q", ["c", "a", "b"], {"a": "", "b": "", "c": ""}) == ["a", "b", "c"]


def test_hf_reranker_empty_candidates():
    calls = {"n": 0}

    def fake(pairs):
        calls["n"] += 1
        return []

    r = HFCrossEncoderReranker(score_batch=fake)
    assert r.rerank("q", [], {}) == []
    assert calls["n"] == 0  # short-circuits before scoring


def test_hf_reranker_missing_passage_scored_as_empty():
    captured = {}

    def fake(pairs):
        captured["pairs"] = pairs
        return [0.0] * len(pairs)

    r = HFCrossEncoderReranker(score_batch=fake)
    r.rerank("q", ["a"], {})  # id 'a' absent from passages
    assert captured["pairs"] == [("q", "")]


def test_hf_reranker_defaults_to_medcpt():
    assert HFCrossEncoderReranker().model_id == "ncbi/MedCPT-Cross-Encoder"
