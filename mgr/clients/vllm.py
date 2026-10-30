"""vLLM generation client: the local-A100 OpenAI-compatible server.

This is where 70B generation runs (AWQ-int4 on one A100, Doc 00 D3). High
concurrency, no free-tier throttle — the rpm cap is left generous and the
binding resource is GPU saturation via continuous batching, not request rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .openai_compat import OpenAICompatClient


def _trailing_label_logprob(prompt_logprobs: Sequence[Mapping[str, Any] | None], label: str) -> float:
    """Sum the logprobs of the trailing prompt tokens that spell ``label``.

    ``prompt_logprobs`` is vLLM's per-prompt-token list (``None`` for the first
    token, else ``{token_id: {"logprob", "rank", "decoded_token"}}``). Walks back
    from the end until the decoded text covers the label. A token that fuses the
    label's first characters with template text (e.g. ``"\\n\\nA"``) still counts:
    that text is identical for every label, so the comparison across labels holds.
    """
    target = label.strip()
    text, total = "", 0.0
    for pos in reversed(list(prompt_logprobs)):
        if not pos:
            break
        fits = []
        for cand in pos.values():
            s = ((cand.get("decoded_token") or "") + text).strip()
            if target.endswith(s) or s.endswith(target):
                fits.append(cand)
        if len(fits) != 1:
            break
        text = (fits[0].get("decoded_token") or "") + text
        total += float(fits[0]["logprob"])
        if text.strip().endswith(target):
            return total
    raise ValueError(f"label {label!r} not found at the end of the scored prompt")


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

    def complete_text_logprobs(
        self, model: str, messages: list[dict[str, str]], *, top_logprobs: int = 10, **params: Any
    ) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
        """Like :meth:`complete_text`, but also request per-token logprobs and
        return the OpenAI ``logprobs.content`` list (each entry carries a
        ``top_logprobs`` candidate list). Drives Gate-A pre-retrieval feature
        capture (confidence/entropy/margin); see docs/gate_a_phase_a.md.

        Any generation client used with ``GateCapture`` must expose this method.
        """
        resp = self.chat(model, messages, logprobs=True, top_logprobs=top_logprobs, **params)
        choice = resp["choices"][0]
        text = choice["message"]["content"]
        usage = resp.get("usage", {})
        tokens = {
            "in": int(usage.get("prompt_tokens", 0)),
            "out": int(usage.get("completion_tokens", 0)),
        }
        content = (choice.get("logprobs") or {}).get("content") or []
        return text, tokens, list(content)

    def score_labels(
        self, model: str, messages: list[dict[str, str]], labels: Sequence[str]
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Teacher-forced ``log P(label | prompt)`` for each answer label (the
        Gate-A probe, paper eq. labelscore).

        Each label is sent as a partial assistant turn (``continue_final_message``)
        and read back from vLLM's ``prompt_logprobs``, so no label has to make a
        top-k list. One request per label; with prefix caching on, only the label
        tokens are new work, which ``in_cached`` shows when the server reports it.
        Returns the logprobs and the token usage summed over the requests.
        """
        scores: dict[str, float] = {}
        tokens = {"in": 0, "out": 0, "in_cached": 0}
        for label in labels:
            msgs = [*messages, {"role": "assistant", "content": label}]
            resp = self.chat(model, msgs, add_generation_prompt=False, continue_final_message=True,
                             prompt_logprobs=0, max_tokens=1, temperature=0.0)
            scores[label] = _trailing_label_logprob(resp.get("prompt_logprobs") or [], label)
            usage = resp.get("usage", {})
            tokens["in"] += int(usage.get("prompt_tokens", 0))
            tokens["out"] += int(usage.get("completion_tokens", 0))
            tokens["in_cached"] += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return scores, tokens
