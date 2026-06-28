"""NIM free-tier client: embeddings / reranker / RAGAS judge / graph extraction.

Throttled to <=40 req/min with exp-backoff + jitter on 429 (Doc 00 section 5).

Hard guard: this client *refuses to route generation*. Free-tier throttling
would wreck the bulk 70B sweep, so generation is local-A100 (vLLM) only. Calling
``chat`` here raises — use :class:`mgr.clients.vllm.VLLMClient` for generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .openai_compat import OpenAICompatClient


class GenerationOnNIMError(RuntimeError):
    """Raised when generation is (mis)routed to the NIM free tier."""


@dataclass
class NimClient:
    base_url: str
    api_key: str | None = None
    rpm: int = 40

    def __post_init__(self) -> None:
        self._client = OpenAICompatClient(base_url=self.base_url, api_key=self.api_key, rpm=self.rpm)

    def embeddings(self, model: str, inputs: list[str], **params: Any) -> dict[str, Any]:
        return self._client.embeddings(model, inputs, **params)

    def judge(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]:
        """RAGAS / LLM-as-judge scoring — a permitted, low-volume NIM use."""
        return self._client.chat(model, messages, **params)

    def rank(self, model: str, query: str, passages: list[str], **params: Any) -> dict[str, Any]:
        """Cross-encoder reranking via the NIM ranking endpoint.

        Returns the raw body, conventionally ``{"rankings": [{"index", "logit"}]}``.
        Payload shape may need adjusting to your specific NIM reranker.
        """
        payload = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            **params,
        }
        return self._client._post("/v1/ranking", payload)

    def chat(self, *_args: Any, **_kwargs: Any):  # noqa: D401 - guard
        raise GenerationOnNIMError(
            "generation must run on local vLLM (A100), not the NIM free tier; "
            "use mgr.clients.vllm.VLLMClient"
        )
