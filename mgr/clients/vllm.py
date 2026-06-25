"""vLLM generation client: the local-A100 OpenAI-compatible server.

This is where 70B generation runs (AWQ-int4 on one A100, Doc 00 D3). High
concurrency, no free-tier throttle — the rpm cap is left generous and the
binding resource is GPU saturation via continuous batching, not request rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .openai_compat import OpenAICompatClient


@dataclass
class VLLMClient:
    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    rpm: int = 100_000  # effectively unthrottled; saturation is handled by the sweep

    def __post_init__(self) -> None:
        self._client = OpenAICompatClient(base_url=self.base_url, api_key=self.api_key, rpm=self.rpm)

    def chat(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]:
        return self._client.chat(model, messages, **params)

    def complete_text(self, model: str, messages: list[dict[str, str]], **params: Any) -> tuple[str, dict[str, int]]:
        """Return (assistant_text, token_usage) from a chat completion."""
        resp = self.chat(model, messages, **params)
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        tokens = {
            "in": int(usage.get("prompt_tokens", 0)),
            "out": int(usage.get("completion_tokens", 0)),
        }
        return text, tokens
