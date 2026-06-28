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

import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class _NimLike(Protocol):
    def embeddings(self, model: str, inputs: list[str], **params: Any) -> dict[str, Any]: ...
    def judge(self, model: str, messages: list[dict[str, str]], **params: Any) -> dict[str, Any]: ...
    def rank(self, model: str, query: str, passages: list[str], **params: Any) -> dict[str, Any]: ...


def _content(resp: dict[str, Any]) -> str:
    return resp["choices"][0]["message"]["content"].strip()


def _is_yes(text: str) -> bool:
    return bool(re.match(r"\s*(yes|true|1)\b", text.strip(), re.I))


def _first_float(text: str, default: float = 0.0) -> float:
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group(0)) if m else default


def _lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:[-*\d.)]+)\s*", "", ln).strip()  # strip bullets/numbering
        if ln:
            out.append(ln)
    return out


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


_JUDGE_SYS = {"role": "system", "content": "You are a meticulous biomedical evaluation judge. Follow the output format exactly."}


@dataclass
class NimGroundingJudge:
    client: _NimLike
    model: str = "meta/llama-3.1-8b-instruct"

    def _ask(self, prompt: str) -> str:
        return _content(self.client.judge(self.model, [_JUDGE_SYS, {"role": "user", "content": prompt}]))

    def decompose(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return _lines(self._ask(
            "Break the following answer into atomic factual claims, one per line:\n\n" + text
        ))

    def entails(self, hypothesis: str, premise: str) -> bool:
        return _is_yes(self._ask(
            f"Premise:\n{premise}\n\nClaim:\n{hypothesis}\n\n"
            "Is the claim fully supported by the premise? Answer yes or no."
        ))

    def relevant(self, question: str, passage: str) -> bool:
        return _is_yes(self._ask(
            f"Question:\n{question}\n\nPassage:\n{passage}\n\n"
            "Is this passage relevant to answering the question? Answer yes or no."
        ))

    def relevance(self, question: str, answer: str) -> float:
        return max(0.0, min(1.0, _first_float(self._ask(
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
            "Rate how relevant the answer is to the question, from 0.0 to 1.0. Reply with only the number."
        ))))


@dataclass
class NimEntityExtractor:
    client: _NimLike
    model: str = "meta/llama-3.1-8b-instruct"

    def extract(self, text: str) -> list[str]:
        if not text.strip():
            return []
        prompt = (
            "List the medical entities (diseases, drugs, procedures, findings) "
            "mentioned in the text, one per line:\n\n" + text
        )
        return _lines(_content(self.client.judge(self.model, [_JUDGE_SYS, {"role": "user", "content": prompt}])))
