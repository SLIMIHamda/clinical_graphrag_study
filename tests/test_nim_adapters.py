import numpy as np

from mgr.clients.nim_adapters import (
    NimEmbedder,
    NimEntityExtractor,
    NimGroundingJudge,
    NimReranker,
)


class FakeNim:
    """Implements the NIM high-level methods with canned, prompt-aware replies."""

    def embeddings(self, model, inputs, **kw):
        # 2-d toy embedding: [len, vowel_count]
        return {"data": [{"embedding": [float(len(t)), float(sum(c in "aeiou" for c in t))]} for t in inputs]}

    def rank(self, model, query, passages, **kw):
        # logit = token overlap with the query
        q = set(query.lower().split())
        return {"rankings": [{"index": i, "logit": len(q & set(p.lower().split()))} for i, p in enumerate(passages)]}

    def judge(self, model, messages, **kw):
        prompt = messages[-1]["content"].lower()
        if "atomic factual claims" in prompt:
            content = "claim one\nclaim two"
        elif "fully supported" in prompt:
            content = "yes" if "aspirin" in prompt else "no"
        elif "relevant to answering" in prompt:
            content = "yes" if "myocardial" in prompt else "no"
        elif "how relevant" in prompt:
            content = "0.8"
        elif "medical entities" in prompt:
            content = "- aspirin\n- myocardial infarction"
        else:
            content = ""
        return {"choices": [{"message": {"content": content}}]}


NIM = FakeNim()


def test_embedder_returns_matrix():
    emb = NimEmbedder(NIM)
    m = emb(["abc", "aeiou"])
    assert isinstance(m, np.ndarray)
    assert m.shape == (2, 2)
    assert m[1, 1] == 5.0  # aeiou has 5 vowels


def test_reranker_orders_by_logit():
    rr = NimReranker(NIM)
    passages = {"d1": "france paris", "d2": "aspirin myocardial infarction"}
    out = rr.rerank("aspirin myocardial infarction", ["d1", "d2"], passages)
    assert out[0] == "d2"


def test_reranker_empty():
    assert NimReranker(NIM).rerank("q", [], {}) == []


def test_judge_decompose_and_entails():
    j = NimGroundingJudge(NIM)
    assert j.decompose("some answer") == ["claim one", "claim two"]
    assert j.entails("aspirin helps", "aspirin reduces clots") is True
    assert j.entails("insulin cures htn", "unrelated text") is False


def test_judge_relevant_and_relevance():
    j = NimGroundingJudge(NIM)
    assert j.relevant("myocardial infarction tx", "passage about myocardial stuff") is True
    assert j.relevance("q", "a") == 0.8


def test_entity_extractor_parses_bullets():
    ex = NimEntityExtractor(NIM)
    assert ex.extract("patient on aspirin after MI") == ["aspirin", "myocardial infarction"]
    assert ex.extract("") == []
