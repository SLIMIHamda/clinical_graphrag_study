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
        self._rank_path: str | None = None  # resolved on first successful rank()

    def embeddings(self, model: str, inputs: list[str], **params: Any) -> dict[str, Any]:
        return self._client.embeddings(model, inputs, **params)

    def judge(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]:
        """RAGAS / LLM-as-judge scoring — a permitted, low-volume NIM use."""
        return self._client.chat(model, messages, **params)

    def rank(self, model: str, query: str, passages: list[str], **params: Any) -> dict[str, Any]:
        """Cross-encoder reranking via the NIM ranking endpoint.

        Returns the raw body, conventionally ``{"rankings": [{"index", "logit"}]}``.

        The two NIM deployments expose reranking at *different* paths, and the
        wrong one answers ``HTTP 404: 404 page not found`` (which the fusion
        retriever then silently degrades past, so the cross-encoder never runs
        and C3 measures a token-overlap fallback instead):

          hosted  build.nvidia.com / integrate.api.nvidia.com
                  -> ``/v1/retrieval/{model}/reranking``
          self-hosted NIM container
                  -> ``/v1/ranking``

        We try the hosted path first, fall back to the container path on 404,
        and remember whichever answered so later calls cost one request.

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
        candidates = [f"/v1/retrieval/{model}/reranking", "/v1/ranking"]
        if self._rank_path is not None:
            candidates = [self._rank_path]

        last: TransportError | None = None
        for path in candidates:
            try:
                body = self._client._post(path, payload)
                self._rank_path = path
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
