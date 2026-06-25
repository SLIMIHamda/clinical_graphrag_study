"""Rate-limited, retry-backed OpenAI-compatible client.

Shared core for both serving backends (Doc 00 section 5):
  - NIM free tier (clients/nim.py) for embeddings / reranker / RAGAS judge,
    throttled to <=40 req/min.
  - local vLLM on the A100 for 70B generation (high concurrency).

The HTTP transport, monotonic clock, and sleep are injectable so the rate
limiter and backoff are unit-testable without a network or real time.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class TransportError(Exception):
    """Wraps a transport failure with its HTTP status (or None)."""

    def __init__(self, status: int | None, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


@dataclass
class Response:
    status: int
    body: dict[str, Any]


class Transport(Protocol):
    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> Response: ...


def _urllib_transport(url: str, headers: dict[str, str], payload: dict[str, Any]) -> Response:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return Response(status=resp.status, body=json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise TransportError(e.code, body) from e
    except urllib.error.URLError as e:
        raise TransportError(None, str(e.reason)) from e


class TokenBucket:
    """A simple token bucket: ``rate`` tokens refilled over ``per_seconds``.

    ``acquire`` blocks (via the injected sleep) until a token is free, so callers
    are paced to at most ``rate`` requests per ``per_seconds`` window.
    """

    def __init__(
        self,
        rate: int,
        per_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rate = rate
        self.per_seconds = per_seconds
        self.capacity = float(rate)
        self._tokens = float(rate)
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * (self.rate / self.per_seconds))

    def acquire(self, n: float = 1.0) -> None:
        while True:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return
            deficit = n - self._tokens
            self._sleep(deficit * (self.per_seconds / self.rate))


@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str | None = None
    rpm: int = 40
    max_retries: int = 6
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 60.0
    transport: Transport = _urllib_transport
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.bucket = TokenBucket(self.rpm, 60.0, clock=self.clock, sleep=self.sleep)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        attempt = 0
        while True:
            self.bucket.acquire()
            try:
                resp = self.transport(url, self._headers(), payload)
                return resp.body
            except TransportError as e:
                retryable = e.status in (429, 500, 502, 503, 504) or e.status is None
                attempt += 1
                if not retryable or attempt > self.max_retries:
                    raise
                # exponential backoff with full jitter
                wait = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
                self.sleep(random.uniform(0, wait))

    def chat(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]:
        return self._post("/v1/chat/completions", {"model": model, "messages": messages, **params})

    def embeddings(self, model: str, inputs: list[str], **params: Any) -> dict[str, Any]:
        return self._post("/v1/embeddings", {"model": model, "input": inputs, **params})
