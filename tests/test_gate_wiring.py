import json
import math

import pytest

from manifest.manifest import load_manifest
from mgr.clients.openai_compat import Response
from mgr.clients.vllm import VLLMClient, _trailing_label_logprob
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


class FakeScoringClient:
    """Generation client with the teacher-forced label-scoring path (the default)."""

    def __init__(self, answer="B", fail=False):
        self.answer = answer
        self.fail = fail
        self.scored: list[list[str]] = []
        self.logprob_calls = 0

    def complete_text(self, model, messages, **params):
        return self.answer, {"in": 50, "out": 1}

    def complete_text_logprobs(self, model, messages, *, top_logprobs=10, **params):
        self.logprob_calls += 1
        return self.answer, {"in": 100, "out": 1}, []

    def score_labels(self, model, messages, labels):
        if self.fail:
            raise ValueError("label 'A' not found")
        self.scored.append(list(labels))
        lp = {lab: math.log(0.05) for lab in labels}
        lp[self.answer] = math.log(0.8)                 # valid mass 0.8 + 3 * 0.05 = 0.95
        return lp, {"in": 40 * len(labels), "out": len(labels), "in_cached": 30 * len(labels)}


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


def _plp(*tokens):
    """vLLM-style prompt_logprobs: None for the first token, then one entry per token."""
    return [None] + [{str(i): {"logprob": lp, "rank": 1, "decoded_token": t}}
                     for i, (t, lp) in enumerate(tokens)]


def test_trailing_label_logprob_spans_split_and_fused_tokens():
    assert _trailing_label_logprob(_plp(("<|eot|>", -0.1), ("A", -0.5)), "A") == -0.5
    # a label tokenized into pieces sums every piece
    assert _trailing_label_logprob(_plp(("\n\n", -0.1), ("may", -1.0), ("be", -0.25)), "maybe") == -1.25
    # a token fusing template whitespace with the label still counts
    assert _trailing_label_logprob(_plp(("assistant", -0.1), ("\n\nB", -2.0)), "B") == -2.0
    with pytest.raises(ValueError, match="not found"):
        _trailing_label_logprob(_plp(("A", -0.5), ("\n", -0.1), ("<|x|>", -0.1)), "A")


def test_client_score_labels_teacher_forces_each_label():
    seen: list[dict] = []

    def fake_transport(url, headers, payload):
        seen.append(payload)
        label = payload["messages"][-1]["content"]
        return Response(status=200, body={
            "choices": [{"message": {"content": "x"}}],
            "prompt_logprobs": _plp(("Options", -3.0), (label, {"A": -0.2, "B": -1.9}[label])),
            "usage": {"prompt_tokens": 20, "completion_tokens": 1,
                      "prompt_tokens_details": {"cached_tokens": 19}},
        })

    vc = VLLMClient(base_url="http://x")
    vc._client.transport = fake_transport
    msgs = [{"role": "user", "content": "Q?"}]
    scores, tokens = vc.score_labels("m", msgs, ["A", "B"])

    assert scores == {"A": -0.2, "B": -1.9}
    assert tokens == {"in": 40, "out": 2, "in_cached": 38}
    p = seen[0]
    assert p["messages"][-1] == {"role": "assistant", "content": "A"} and len(p["messages"]) == 2
    assert p["continue_final_message"] is True and p["add_generation_prompt"] is False
    assert p["prompt_logprobs"] == 0 and p["max_tokens"] == 1


def test_norag_capture_scores_labels_by_default(fixture_data, tmp_path):
    client = FakeScoringClient(answer="B")
    execu = RAGExecutor(client=client, data_root=fixture_data, n_items=10,
                        gate=GateCapture(enabled=True, n_samples=2))
    rec = _run_r0001(fixture_data, tmp_path / "results", execu)
    assert rec is not None and rec.status == "Done"

    r0 = _items(tmp_path / "results")[0]
    assert client.logprob_calls == 0                       # generation took the plain path
    assert client.scored[0] == ["A", "B", "C", "D"]         # every option label scored
    assert r0["label_scoring"] == "teacher_forced" and r0["probe_answer"] == "B"
    assert abs(r0["valid_mass"] - 0.95) < 1e-9
    assert abs(r0["option_probs"]["B"] - 0.8 / 0.95) < 1e-5
    assert abs(r0["confidence"] - 0.8 / 0.95) < 1e-9 and r0["n_options"] == 4.0
    assert r0["sc_matches_greedy"] == 1.0                   # samples agree with the probe answer
    # probe cost = label scoring (4 x 40 in) + 2 samples (2 x 50 in), off the arm's total
    assert r0["gate_cost_tokens"] == {"in": 260, "out": 6, "label_in_cached": 120}
    assert r0["tokens"] == {"in": 50, "out": 1}


def test_failed_probe_marks_the_item_and_keeps_the_run(fixture_data, tmp_path):
    execu = RAGExecutor(client=FakeScoringClient(fail=True), data_root=fixture_data, n_items=10,
                        gate=GateCapture(enabled=True, n_samples=0))
    rec = _run_r0001(fixture_data, tmp_path / "results", execu)
    assert rec is not None and rec.status == "Done"
    r0 = _items(tmp_path / "results")[0]
    assert r0["gate_error"].startswith("ValueError") and "confidence" not in r0
    assert "no_rag_correct" not in r0


def test_norag_capture_enriches_item_rows(fixture_data, tmp_path):
    client = FakeLogprobClient(answer="B")
    gate = GateCapture(enabled=True, label_scoring="topk", n_samples=3,
                       embed_client=FakeEmbedClient(), embed_model="fake")
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


def test_capture_requires_scoring_client():
    execu = RAGExecutor(client=FakeLogprobClient(), data_root=".", gate=GateCapture(enabled=True))
    with pytest.raises(RuntimeError, match="score_labels"):
        execu(None, {})  # guard fires before any cfg/row access


def test_topk_capture_requires_logprob_client():
    execu = RAGExecutor(client=FakeNoLogprobClient(), data_root=".",
                        gate=GateCapture(enabled=True, label_scoring="topk"))
    with pytest.raises(RuntimeError, match="complete_text_logprobs"):
        execu(None, {})


def test_unknown_label_scoring_mode_is_rejected():
    execu = RAGExecutor(client=FakeScoringClient(), data_root=".",
                        gate=GateCapture(enabled=True, label_scoring="first_char"))
    with pytest.raises(ValueError, match="label_scoring"):
        execu(None, {})
