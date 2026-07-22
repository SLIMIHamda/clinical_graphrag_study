import pytest

from mgr.clients.nim import GenerationOnNIMError, NimClient
from mgr.clients.openai_compat import (
    OpenAICompatClient,
    Response,
    TokenBucket,
    TransportError,
)
from mgr.clients.vllm import VLLMClient


class FakeClock:
    """Deterministic clock + sleep: sleeping advances virtual time."""

    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.slept += s


def test_token_bucket_paces_after_capacity():
    clk = FakeClock()
    tb = TokenBucket(rate=40, per_seconds=60.0, clock=clk.now, sleep=clk.sleep)
    # first 40 are immediate (full bucket), no sleep
    for _ in range(40):
        tb.acquire()
    assert clk.slept == 0.0
    # the 41st must wait ~ one refill interval (60/40 = 1.5s)
    tb.acquire()
    assert clk.slept == pytest.approx(1.5, rel=1e-6)


def test_backoff_retries_then_succeeds():
    clk = FakeClock()
    calls = {"n": 0}

    def flaky_transport(url, headers, payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransportError(429, "rate limited")
        return Response(200, {"ok": True})

    c = OpenAICompatClient(
        base_url="http://x", transport=flaky_transport, sleep=clk.sleep, clock=clk.now, rpm=1000
    )
    out = c.chat("m", [{"role": "user", "content": "hi"}])
    assert out == {"ok": True}
    assert calls["n"] == 3  # two failures + one success


def test_backoff_gives_up_after_max_retries():
    def always_429(url, headers, payload):
        raise TransportError(429, "nope")

    c = OpenAICompatClient(
        base_url="http://x", transport=always_429, sleep=lambda s: None, rpm=1000, max_retries=3
    )
    with pytest.raises(TransportError) as e:
        c.chat("m", [{"role": "user", "content": "hi"}])
    assert e.value.status == 429


def test_non_retryable_status_raises_immediately():
    calls = {"n": 0}

    def bad_request(url, headers, payload):
        calls["n"] += 1
        raise TransportError(400, "bad")

    c = OpenAICompatClient(base_url="http://x", transport=bad_request, sleep=lambda s: None, rpm=1000)
    with pytest.raises(TransportError):
        c.embeddings("e", ["a"])
    assert calls["n"] == 1  # 400 is not retried


def test_nim_refuses_generation():
    nim = NimClient(base_url="http://nim", api_key="k")
    with pytest.raises(GenerationOnNIMError):
        nim.chat("model", [{"role": "user", "content": "x"}])


def _rank_client(serving_path: str):
    """A NimClient whose fake transport only answers on ``serving_path``."""
    seen: list[str] = []

    def transport(url, headers, payload):
        seen.append(url)
        if url.endswith(serving_path):
            return Response(200, {"rankings": [{"index": 0, "logit": 1.0}]})
        raise TransportError(404, "404 page not found")

    nim = NimClient(base_url="http://nim", api_key="k")
    nim._client = OpenAICompatClient(
        base_url="http://nim", api_key="k", transport=transport, sleep=lambda s: None, rpm=1000
    )
    return nim, seen


def test_rank_uses_hosted_retrieval_path():
    nim, seen = _rank_client("/v1/retrieval/some/model/reranking")
    out = nim.rank("some/model", "q", ["p"])
    assert out["rankings"][0]["index"] == 0
    assert seen == ["http://nim/v1/retrieval/some/model/reranking"]


def test_rank_falls_back_to_container_path_on_404():
    """A hosted-vs-self-hosted path mismatch used to surface as a bare 404,
    which the fusion retriever swallowed — so C3 silently measured a
    token-overlap fallback instead of the cross-encoder."""
    nim, seen = _rank_client("/v1/ranking")
    out = nim.rank("some/model", "q", ["p"])
    assert out["rankings"][0]["index"] == 0
    assert seen[-1] == "http://nim/v1/ranking"
    # and the resolved path is remembered, so the next call costs one request
    nim.rank("some/model", "q", ["p"])
    assert seen[-2:] == ["http://nim/v1/ranking", "http://nim/v1/ranking"]


def test_rank_truncates_by_default():
    """`truncate=END` keeps an over-long passage from returning HTTP 400."""
    captured = {}

    def transport(url, headers, payload):
        captured.update(payload)
        return Response(200, {"rankings": []})

    nim = NimClient(base_url="http://nim", api_key="k")
    nim._client = OpenAICompatClient(
        base_url="http://nim", transport=transport, sleep=lambda s: None, rpm=1000
    )
    nim.rank("m", "q", ["p"])
    assert captured["truncate"] == "END"


def test_rank_does_not_retry_a_real_error():
    """Only 404 triggers the path probe; anything else propagates."""

    def transport(url, headers, payload):
        raise TransportError(401, "unauthorized")

    nim = NimClient(base_url="http://nim", api_key="k")
    nim._client = OpenAICompatClient(
        base_url="http://nim", transport=transport, sleep=lambda s: None, rpm=1000
    )
    with pytest.raises(TransportError) as e:
        nim.rank("m", "q", ["p"])
    assert e.value.status == 401


def test_vllm_complete_text_parses_usage():
    def fake(url, headers, payload):
        return Response(
            200,
            {
                "choices": [{"message": {"content": "B"}}],
                "usage": {"prompt_tokens": 1234, "completion_tokens": 7},
            },
        )

    v = VLLMClient(base_url="http://vllm")
    v._client.transport = fake
    text, tokens = v.complete_text("llama", [{"role": "user", "content": "q"}])
    assert text == "B"
    assert tokens == {"in": 1234, "out": 7}
