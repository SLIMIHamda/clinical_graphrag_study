"""NIM free-tier client: embeddings / reranker / RAGAS judge / graph extraction.

Throttled to <=40 req/min with exp-backoff + jitter on 429 (Doc 00 section 5).

Hard guard: this client *refuses to route generation*. Free-tier throttling
would wreck the bulk 70B sweep, so generation is local-A100 (vLLM) only. Calling
``chat`` here raises — use :class:`mgr.clients.vllm.VLLMClient` for generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .openai_compat import OpenAICompatClient, TransportError


class GenerationOnNIMError(RuntimeError):
    """Raised when generation is (mis)routed to the NIM free tier."""


@dataclass
class NimClient:
    base_url: str
    api_key: str | None = None
    rpm: int = 40

    def __post_init__(self) -> None:
        self._client = OpenAICompatClient(base_url=self.base_url, api_key=self.api_key, rpm=self.rpm)
        self._rank_endpoint: tuple[str, str] | None = None  # (base_url, path), resolved on first rank()
        self._extra_clients: dict[str, OpenAICompatClient] = {}

    def _client_for(self, base_url: str) -> OpenAICompatClient:
        """Reuse the primary client for its own host; lazily build one per extra host.

        Reranking is low-volume and, after the first success, only one endpoint is
        ever used (it is cached), so the separate token buckets do not race in
        practice — the probe costs at most a couple of extra requests once.
        """
        if base_url.rstrip("/") == self.base_url.rstrip("/"):
            return self._client
        if base_url not in self._extra_clients:
            self._extra_clients[base_url] = OpenAICompatClient(
                base_url=base_url, api_key=self.api_key, rpm=self.rpm
            )
        return self._extra_clients[base_url]

    def embeddings(self, model: str, inputs: list[str], **params: Any) -> dict[str, Any]:
        return self._client.embeddings(model, inputs, **params)

    def judge(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]:
        """RAGAS / LLM-as-judge scoring — a permitted, low-volume NIM use."""
        return self._client.chat(model, messages, **params)

    def rank(self, model: str, query: str, passages: list[str], **params: Any) -> dict[str, Any]:
        """Cross-encoder reranking via the NIM ranking endpoint.

        Returns the raw body, conventionally ``{"rankings": [{"index", "logit"}]}``.

        NVIDIA exposes reranking for these QA models at *different hosts and
        paths*, and hitting the wrong one answers ``HTTP 404: 404 page not
        found`` (a Go-gateway default). The fusion retriever silently degrades
        past that to a token-overlap fallback, so the cross-encoder never runs
        and C3 measures the fallback instead — which is exactly what happened in
        the 2026-07 POC runs, where ``integrate.api.nvidia.com/v1/ranking`` 404'd
        for ``nv-rerankqa-mistral-4b-v3``. The documented endpoints are:

          OpenAI-style gateway / self-hosted container
                  -> ``{base_url}/v1/ranking``
          hosted retrieval NIM (the QA rerank models live here)
                  -> ``https://ai.api.nvidia.com/v1/retrieval/{model}/reranking``

        We try each once, fall through to the next on 404, and cache whichever
        answered so later calls cost a single request. The request/response
        contract (``query.text`` / ``passages[].text`` -> ``rankings[]``) is the
        same across hosts, so the payload is shared.

        ``truncate="END"`` is sent by default so an over-long passage is clipped
        server-side rather than returning ``HTTP 400: Input length ... exceeds
        maximum allowed token size``.
        """
        payload = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            **{"truncate": "END", **params},
        }
        base = self.base_url.rstrip("/")
        # (base_url, path) candidates, most-likely-correct first to minimise probes.
        candidates: list[tuple[str, str]] = [
            (base, "/v1/ranking"),
            ("https://ai.api.nvidia.com", f"/v1/retrieval/{model}/reranking"),
            (base, f"/v1/retrieval/{model}/reranking"),
        ]
        if self._rank_endpoint is not None:
            candidates = [self._rank_endpoint]

        last: TransportError | None = None
        for cand_base, path in candidates:
            try:
                body = self._client_for(cand_base)._post(path, payload)
                self._rank_endpoint = (cand_base, path)
                return body
            except TransportError as e:
                if e.status != 404:
                    raise
                last = e
        raise last if last is not None else TransportError(404, "no reranking endpoint resolved")

    def chat(self, *_args: Any, **_kwargs: Any):  # noqa: D401 - guard
        raise GenerationOnNIMError(
            "generation must run on local vLLM (A100), not the NIM free tier; "
            "use mgr.clients.vllm.VLLMClient"
        )
