import numpy as np
import pytest

from mgr.retrieval.dense import DenseIndex, DenseRetriever
from mgr.retrieval.factory import build_retriever

VOCAB = ["aspirin", "mi", "reperfusion", "diabetes", "france"]


def _embed(texts):
    """Toy bag-of-vocab embedder: vector of term counts over VOCAB."""
    out = np.zeros((len(texts), len(VOCAB)), dtype=float)
    for i, t in enumerate(texts):
        toks = t.lower().split()
        for j, w in enumerate(VOCAB):
            out[i, j] = toks.count(w)
    return out


def _corpus():
    return [
        {"id": "d1", "text": "aspirin mi"},
        {"id": "d2", "text": "diabetes"},
        {"id": "d3", "text": "aspirin mi reperfusion"},
        {"id": "d4", "text": "france"},
    ]


def test_dense_index_ranks_by_cosine():
    idx = DenseIndex.from_corpus(_corpus(), _embed)
    qv = _embed(["aspirin mi"])[0]
    hits = idx.search(qv, top_k=3)
    ids = [h[0] for h in hits]
    assert ids[0] in {"d1", "d3"}    # both contain aspirin+mi
    assert "d4" not in ids           # france is orthogonal
    assert hits[0][1] == pytest.approx(1.0, abs=1e-9)  # cosine 1.0 for exact match


def test_dense_from_precomputed_embeddings():
    records = _corpus()
    matrix = _embed([r["text"] for r in records])
    idx = DenseIndex.from_embeddings([r["id"] for r in records], matrix)
    hits = idx.search(_embed(["diabetes"])[0], top_k=1)
    assert hits[0][0] == "d2"


def test_dense_retriever_builds_context_and_ranks():
    records = _corpus()
    passages = {r["id"]: r["text"] for r in records}
    idx = DenseIndex.from_corpus(records, _embed)
    r = DenseRetriever(idx, _embed, passages)
    res = r.retrieve("aspirin mi reperfusion", depth_k=2)
    assert res.context is not None
    assert res.retrieved_ids[0] == "d3"      # best full match
    assert res.ranks[res.retrieved_ids[0]] == 1


def test_index_rejects_misaligned_matrix():
    with pytest.raises(ValueError):
        DenseIndex(["a", "b"], np.zeros((3, 4)))


def test_factory_wires_dense_with_precomputed():
    records = _corpus()
    matrix = _embed([r["text"] for r in records])
    passages = {r["id"]: r["text"] for r in records}
    retr = build_retriever(
        "Dense-MedCPT",
        embedder=_embed,
        embeddings=matrix,
        doc_ids=[r["id"] for r in records],
        passages=passages,
    )
    res = retr.retrieve("diabetes", depth_k=1)
    assert res.retrieved_ids == ["d2"]


def test_factory_dense_requires_embedder():
    with pytest.raises(ValueError):
        build_retriever("Dense-MedCPT", _corpus())
