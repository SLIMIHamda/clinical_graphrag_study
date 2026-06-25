from mgr.retrieval.base import NullRetriever
from mgr.retrieval.bm25 import BM25Index, BM25Retriever, Doc


def _corpus():
    return [
        Doc("d1", "myocardial infarction is treated with aspirin and reperfusion"),
        Doc("d2", "diabetes mellitus management includes metformin and insulin"),
        Doc("d3", "aspirin reduces platelet aggregation in acute coronary syndrome"),
        Doc("d4", "the capital of france is paris"),
    ]


def test_bm25_ranks_relevant_docs_first():
    idx = BM25Index(_corpus())
    hits = idx.search("aspirin for myocardial infarction", top_k=3)
    ids = [h[0] for h in hits]
    assert ids[0] in {"d1", "d3"}  # aspirin/MI docs beat france/diabetes
    assert "d4" not in ids


def test_bm25_retriever_builds_context_and_ranks():
    r = BM25Retriever(BM25Index(_corpus()))
    res = r.retrieve("aspirin coronary syndrome", depth_k=2)
    assert res.context is not None
    assert len(res.retrieved_ids) <= 2
    assert res.ranks[res.retrieved_ids[0]] == 1
    assert res.fused_scores[0] >= res.fused_scores[-1]


def test_bm25_no_match_returns_empty_context():
    r = BM25Retriever(BM25Index(_corpus()))
    res = r.retrieve("xyzzy nonexistent term", depth_k=5)
    assert res.context is None
    assert res.retrieved_ids == []


def test_null_retriever_is_closed_book():
    res = NullRetriever().retrieve("anything")
    assert res.context is None
    assert res.retrieved_ids == []


def test_context_char_cap_truncates():
    # chunk-sized passages (~250 chars); cap admits only the first few.
    big = [Doc(f"d{i}", "word " * 50) for i in range(10)]
    r = BM25Retriever(BM25Index(big), max_context_chars=1000)
    res = r.retrieve("word", depth_k=10)
    assert res.context is not None
    assert len(res.context) <= 1000 + 50      # cap plus id-prefix margin
    assert len(res.retrieved_ids) == 10       # all ranked...
    assert res.context.count("[d") < 10       # ...but not all fit in context
