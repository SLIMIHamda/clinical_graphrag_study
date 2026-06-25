"""BM25 (Okapi) lexical retrieval.

Pure-Python Okapi BM25 over an in-memory corpus — enough for the 200-item smoke
and unit tests. At full scale the same :class:`Retriever` interface wraps the
reused MedRAG precomputed indices (PubMed/StatPearls/Textbooks/Wikipedia), so
the executor and prompt path never change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .base import RetrievalResult


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Doc:
    doc_id: str
    text: str


class BM25Index:
    """Okapi BM25 with the standard k1/b parameters."""

    def __init__(self, docs: list[Doc], *, k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(d.text) for d in docs]
        self._len = [len(t) for t in self._tokens]
        self._avglen = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._tf = [Counter(t) for t in self._tokens]
        # document frequency per term
        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = len(docs)
        # BM25 idf with +1 smoothing (always positive)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    @classmethod
    def from_records(cls, records: list[dict], *, id_key: str = "id", text_key: str = "text", **kw) -> "BM25Index":
        docs = [Doc(doc_id=str(r[id_key]), text=str(r[text_key])) for r in records]
        return cls(docs, **kw)

    def _score(self, q_terms: list[str], i: int) -> float:
        tf, dl = self._tf[i], self._len[i]
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1.0))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q = tokenize(query)
        scored = [(self.docs[i].doc_id, self._score(q, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


@dataclass
class BM25Retriever:
    index: BM25Index
    max_context_chars: int = 8000

    def retrieve(self, query: str, *, depth_k: int = 10) -> RetrievalResult:
        hits = self.index.search(query, top_k=depth_k)
        by_id = {d.doc_id: d.text for d in self.index.docs}
        ids = [h[0] for h in hits]
        passages, used = [], 0
        for did in ids:
            txt = by_id[did]
            if used + len(txt) > self.max_context_chars:
                break
            passages.append(f"[{did}] {txt}")
            used += len(txt)
        context = "\n\n".join(passages) if passages else None
        return RetrievalResult(
            context=context,
            retrieved_ids=ids,
            ranks={did: r + 1 for r, did in enumerate(ids)},
            fused_scores=[h[1] for h in hits],
            rerank_fired=False,
        )
