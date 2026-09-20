"""In-process judge and entity extractor — the offline half of the adapters.

Mirrors :mod:`mgr.clients.nim_adapters` over a local chat client (anything with
``complete_text(model, messages, **params) -> (text, usage)``, which is
:class:`mgr.clients.local_hf.LocalHFClient` and :class:`mgr.clients.vllm.VLLMClient`).
Both judges share prompts and parsers via :mod:`mgr.clients.judge_prompts`, so a
NIM-judged run and a locally-judged run answer the same question in the same
words and their RAGAS columns stay comparable.

Why this is required rather than nice to have:

* **Governance.** RAGAS faithfulness and context precision feed the *answer and
  the retrieved passages* to the judge. On MIMIC those passages are chart text,
  so the hosted judge is not an option and grounding metrics would otherwise be
  unavailable on exactly the benchmark the paper turns on.
* **Graph building.** :class:`LocalEntityExtractor` is the LLM alternative to
  :class:`mgr.graph.extract.NgramExtractor` for corpus-side extraction, for
  vocabulary the dictionary misses.

One deliberate difference from the hosted path: every call sets its own output
budget from ``MAX_TOKENS``. The local client is configured for MCQ answers
(``max_new_tokens=8``) and a judge that inherits that returns a truncated claim
list — which is how the first POC reported faithfulness 0.15 from answers cut to
eight tokens.

Token spend is accumulated on ``usage_tokens`` so the judging cost is reportable
separately from the arm's generation cost rather than disappearing into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .judge_prompts import (
    MAX_TOKENS,
    clamp01,
    decompose_prompt,
    entails_prompt,
    entities_prompt,
    messages_for,
    parse_float,
    parse_lines,
    parse_yes,
    relevance_prompt,
    relevant_prompt,
)


class _ChatLike(Protocol):
    def complete_text(
        self, model: str, messages: list[dict[str, str]], **params: Any
    ) -> tuple[str, dict[str, int]]: ...


@dataclass
class _LocalCaller:
    """Shared plumbing: fixed decoding, per-kind budget, token accounting."""

    client: _ChatLike
    model: str = ""
    temperature: float = 0.0
    usage_tokens: dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0})

    def _ask(self, prompt: str, kind: str) -> str:
        text, usage = self.client.complete_text(
            self.model or getattr(self.client, "model_id", ""),
            messages_for(prompt),
            temperature=self.temperature,
            max_tokens=MAX_TOKENS[kind],
        )
        for k in ("in", "out"):
            self.usage_tokens[k] += int((usage or {}).get(k, 0))
        return (text or "").strip()


@dataclass
class LocalGroundingJudge(_LocalCaller):
    """Satisfies :class:`mgr.metrics.grounding_ragas.GroundingJudge`, locally."""

    def decompose(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return parse_lines(self._ask(decompose_prompt(text), "decompose"))

    def entails(self, hypothesis: str, premise: str) -> bool:
        return parse_yes(self._ask(entails_prompt(hypothesis, premise), "entails"))

    def relevant(self, question: str, passage: str) -> bool:
        return parse_yes(self._ask(relevant_prompt(question, passage), "relevant"))

    def relevance(self, question: str, answer: str) -> float:
        return clamp01(parse_float(self._ask(relevance_prompt(question, answer), "relevance")))


@dataclass
class LocalEntityExtractor(_LocalCaller):
    """Satisfies :class:`mgr.graph.build.Extractor`, locally."""

    def extract(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return parse_lines(self._ask(entities_prompt(text), "entities"))
