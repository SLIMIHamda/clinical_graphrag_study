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
