import numpy as np
import pytest

from mgr.clients.local_embed import MEDCPT_ARTICLE, MEDCPT_QUERY, LocalEmbedder, medcpt_pair
from mgr.retrieval.dense import DenseIndex, DenseRetriever


def fake_encoder(dim=4, calls=None):
    """Deterministic stand-in: vector is (len, first-char ordinal, 1, 0)."""

    def encode(texts):
        if calls is not None:
            calls.append(list(texts))
        return np.array(
            [[len(t), ord(t[0]) if t else 0.0, 1.0, 0.0][:dim] for t in texts], dtype=float
        )

    return encode


def test_returns_one_row_per_text():
    emb = LocalEmbedder(encode_batch=fake_encoder())
    out = emb(["alpha", "beta", "gamma"])
    assert out.shape == (3, 4)


def test_batches_respect_batch_size():
    calls: list[list[str]] = []
    emb = LocalEmbedder(batch_size=2, encode_batch=fake_encoder(calls=calls))
    emb([f"t{i}" for i in range(5)])
    assert [len(c) for c in calls] == [2, 2, 1]


def test_output_is_l2_normalized():
    emb = LocalEmbedder(encode_batch=fake_encoder())
    out = emb(["alpha", "beta"])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_normalize_can_be_disabled():
    emb = LocalEmbedder(normalize=False, encode_batch=fake_encoder())
    out = emb(["alpha"])
    assert np.linalg.norm(out[0]) > 1.0


def test_empty_input_does_not_call_the_model():
    calls: list[list[str]] = []
    emb = LocalEmbedder(encode_batch=fake_encoder(calls=calls))
    assert emb([]).shape == (0, 0)
    assert calls == []


def test_rejects_unknown_pooling():
    with pytest.raises(ValueError):
        LocalEmbedder(pooling="max")


def test_load_is_a_noop_when_encode_batch_is_injected():
    # guards the lazy-import contract: no torch/transformers needed in the test env
    emb = LocalEmbedder(encode_batch=fake_encoder())
    assert emb.load() is emb


def test_medcpt_pair_is_asymmetric():
    passage, query = medcpt_pair()
    assert passage.model_id == MEDCPT_ARTICLE
    assert query.model_id == MEDCPT_QUERY
    assert passage.model_id != query.model_id


def test_satisfies_the_embedder_contract_end_to_end():
    """The point of the class: drop into DenseIndex/DenseRetriever unchanged."""
    emb = LocalEmbedder(encode_batch=fake_encoder())
    records = [{"id": "d1", "text": "aaaa"}, {"id": "d2", "text": "bbbbbbbb"}]
    index = DenseIndex.from_corpus(records, emb)
    assert index.doc_ids == ["d1", "d2"]

    passages = {r["id"]: r["text"] for r in records}
    result = DenseRetriever(index, emb, passages).retrieve("bbbbbbbb", depth_k=1)
    assert result.retrieved_ids == ["d2"]
