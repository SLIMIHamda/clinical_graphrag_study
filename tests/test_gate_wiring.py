import json

import pytest

from manifest.manifest import load_manifest
from mgr.clients.openai_compat import Response
from mgr.clients.vllm import VLLMClient
from mgr.data.loader import write_items_fixture
from mgr.generate.executor import GateCapture, RAGExecutor
from mgr.retrieval.base import RetrievalResult
from mgr.runner import Runner


class FakeLogprobClient:
    """Generation client with the logprob path Gate-A capture needs."""

    def __init__(self, answer="B"):
        self.answer = answer
        self.samples = 0

    def complete_text(self, model, messages, **params):
        self.samples += 1  # used for self-consistency draws
        return self.answer, {"in": 50, "out": 1}

    def complete_text_logprobs(self, model, messages, *, top_logprobs=10, **params):
        content = [{
            "token": self.answer, "logprob": -0.2,
            "top_logprobs": [
                {"token": self.answer, "logprob": -0.2},
                {"token": "A", "logprob": -1.6},
                {"token": "C", "logprob": -2.3},
                {"token": "D", "logprob": -3.0},
            ],
        }]
        return self.answer, {"in": 100, "out": 1}, content


class FakeNoLogprobClient:
    def complete_text(self, model, messages, **params):
        return "B", {"in": 100, "out": 1}


class FakeEmbedClient:
    def embeddings(self, model, inputs, **params):
        return {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]} for _ in inputs]}


class FakeRetriever:
    def retrieve(self, query, depth_k=10):
        return RetrievalResult(context="some passage", retrieved_ids=["p1"])


@pytest.fixture
def fixture_data(tmp_path):
    rows = [
        {"qid": f"mmlu_{i:03d}", "question": f"Q{i}?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
         "answer": "B" if i % 2 == 0 else "C"}
        for i in range(10)
    ]
    write_items_fixture("MMLU-Med", tmp_path, rows)
    return tmp_path


def _items(results_root):
    files = list(results_root.rglob("items.jsonl"))
    assert files, "no items.jsonl written"
    return [json.loads(ln) for ln in files[0].read_text(encoding="utf-8").splitlines() if ln.strip()]


def _run_r0001(fixture_data, results_root, executor):
    runner = Runner(
        manifest=load_manifest(),
        gate_ledger={"H2": True, "G3": False, "P3": False},
        results_root=results_root,
    )
    return runner.run_one("R0001", executor=executor)  # R0001 = No-RAG / MMLU-Med / s42


# --------------------------------------------------------------------------- #
def test_client_complete_text_logprobs_requests_and_parses():
    seen: dict = {}

    def fake_transport(url, headers, payload):
        seen.update(payload)
        return Response(status=200, body={
            "choices": [{
                "message": {"content": "B"},
                "logprobs": {"content": [{
                    "token": "B", "logprob": -0.1,
                    "top_logprobs": [{"token": "B", "logprob": -0.1}, {"token": "A", "logprob": -2.0}],
                }]},
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        })

    vc = VLLMClient(base_url="http://x")
    vc._client.transport = fake_transport
    text, tokens, content = vc.complete_text_logprobs("m", [{"role": "user", "content": "hi"}], top_logprobs=5)

    assert text == "B" and tokens == {"in": 12, "out": 1}
    assert content[0]["top_logprobs"][0]["token"] == "B"
    assert seen["logprobs"] is True and seen["top_logprobs"] == 5  # the request opted in


def test_norag_capture_enriches_item_rows(fixture_data, tmp_path):
    client = FakeLogprobClient(answer="B")
    gate = GateCapture(enabled=True, n_samples=3, embed_client=FakeEmbedClient(), embed_model="fake")
    execu = RAGExecutor(client=client, data_root=fixture_data, n_items=10, gate=gate)  # NullRetriever by default
    rec = _run_r0001(fixture_data, tmp_path / "results", execu)
    assert rec is not None and rec.status == "Done"

    rows = _items(tmp_path / "results")
    r0 = rows[0]
    for key in ("confidence", "entropy", "margin", "sc_agreement", "sc_entropy",
                "sc_matches_greedy", "q_len_words", "n_options", "q_emb", "no_rag_correct", "gate_cost_tokens"):
        assert key in r0, f"missing feature {key}"
    assert r0["n_options"] == 4.0
    assert r0["confidence"] > 0.5 and r0["margin"] > 0.0     # 'B' dominates the option logprobs
    assert r0["sc_agreement"] == 1.0                          # 3 identical 'B' samples
    assert r0["q_emb"] == [0.1, 0.2, 0.3, 0.4]
    assert client.samples == 30                              # 3 self-consistency draws x 10 items


def test_capture_skipped_on_retrieval_arm(fixture_data, tmp_path):
    gate = GateCapture(enabled=True, n_samples=3)
    execu = RAGExecutor(client=FakeLogprobClient(), data_root=fixture_data, n_items=10,
                        retriever=FakeRetriever(), gate=gate)  # non-null retriever -> no capture
    _run_r0001(fixture_data, tmp_path / "results", execu)
    assert "confidence" not in _items(tmp_path / "results")[0]


def test_capture_requires_logprob_client():
    execu = RAGExecutor(client=FakeNoLogprobClient(), data_root=".", gate=GateCapture(enabled=True))
    with pytest.raises(RuntimeError, match="complete_text_logprobs"):
        execu(None, {})  # guard fires before any cfg/row access
