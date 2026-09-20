"""NIM-backed adapters that satisfy the pipeline's injectable interfaces.

Each adapter wraps a NIM client and implements an interface the rest of the code
already consumes, so dropping in a real NIM key activates the real services with
no further code change:

  NimEmbedder        -> Embedder         (dense index / query embedding)
  NimReranker        -> reranker.rerank  (CrossEncoder slot in FusionRetriever)
  NimGroundingJudge  -> GroundingJudge   (RAGAS metrics)
  NimEntityExtractor -> mentions          (UMLS grounding / graph extraction feed)

All NIM calls are embeddings/reranking/judging/extraction — never generation
(the NimClient.chat guard enforces that). Adapters depend only on the client's
high-level methods, so tests inject a fake client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .judge_prompts import (
    JUDGE_SYSTEM,
    MAX_TOKENS,
    clamp01,
    decompose_prompt,
    entails_prompt,
    entities_prompt,
    parse_float,
    parse_lines,
    parse_yes,
    relevance_prompt,
    relevant_prompt,
)


class _NimLike(Protocol):
    def embeddings(self, model: str, inputs: list[str], **params: Any) -> dict[str, Any]: ...
    def judge(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]: ...
    def rank(self, model: str, query: str, passages: list[str], **params: Any) -> dict[str, Any]: ...


def _content(resp: dict[str, Any]) -> str:
    return resp["choices"][0]["message"]["content"].strip()


@dataclass
class NimEmbedder:
    client: _NimLike
    model: str = "nvidia/nv-embedqa-e5-v5"

    def __call__(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings(self.model, list(texts))
        return np.array([d["embedding"] for d in resp["data"]], dtype=float)


@dataclass
class NimReranker:
    client: _NimLike
    model: str = "nvidia/nv-rerankqa-mistral-4b-v3"

    def rerank(self, query: str, candidate_ids: list[str], passages: dict[str, str]) -> list[str]:
        if not candidate_ids:
            return []
        texts = [passages.get(c, "") for c in candidate_ids]
        resp = self.client.rank(self.model, query, texts)
        rankings = sorted(resp["rankings"], key=lambda r: -r["logit"])
        return [candidate_ids[r["index"]] for r in rankings]


@dataclass
class NimGroundingJudge:
    client: _NimLike
    model: str = "meta/llama-3.1-8b-instruct"

    def _ask(self, prompt: str, kind: str) -> str:
        return _content(self.client.judge(
            self.model, [JUDGE_SYSTEM, {"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS[kind],
        ))

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
class NimEntityExtractor:
    client: _NimLike
    model: str = "meta/llama-3.1-8b-instruct"

    def extract(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return parse_lines(_content(self.client.judge(
            self.model, [JUDGE_SYSTEM, {"role": "user", "content": entities_prompt(text)}],
            max_tokens=MAX_TOKENS["entities"],
        )))
