"""Dictionary-driven mention extraction — no model, no API call.

:func:`mgr.graph.build.query_concept_fn` needs an ``Extractor`` (``extract(text)
-> [mention]``) at *query* time, not just at build time. The only implementation
so far is :class:`mgr.clients.nim_adapters.NimEntityExtractor`, which means one
LLM call per query and, under the PhysioNet DUA, chart text leaving the machine.
Neither is acceptable for the local runs, so this is the offline alternative.

It is also the more honest instrument for what the concept layer is supposed to
measure. The first POC linked **1 concept across 60 chunks**, so CA-RRF reordered
the top-5 on 0 of 16 queries — C2 and C3 were measuring an empty feature. A
dictionary scan over a real UMLS term list is deterministic, costs nothing, and
either populates that layer or shows plainly that it cannot be populated.

Two modes:

* **Vocabulary-bounded** (``vocab`` given, the fast path) — a greedy
  longest-match scan, so "heart failure" wins over "heart". Emits only spans the
  dictionary actually contains: O(tokens x max_n) set lookups.
* **Unbounded** (no ``vocab``) — every n-gram up to ``max_n``, left for the
  linker to resolve. Only usable with ``UMLSLinker(fuzzy_enabled=False)``: fuzzy
  linking is a ``difflib`` scan over every key, so an unfiltered candidate list
  against a full-size dictionary does not terminate in useful time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .umls import normalize_term

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")

#: Tokens never worth emitting alone; they still appear inside longer spans.
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his in is it its of on or "
    "s she that the their them they this to was were will with you your".split()
)


@dataclass
class NgramExtractor:
    """Emit candidate mentions from raw text.

    ``vocab`` holds surface forms; they are normalized with the linker's own
    :func:`mgr.graph.umls.normalize_term` on construction, so the caller can pass
    a term -> CUI dictionary's keys verbatim.
    """

    vocab: Iterable[str] | None = None
    max_n: int = 4
    drop_stopwords: bool = True
    _vocab: frozenset[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vocab is not None:
            self._vocab = frozenset(
                n for n in (normalize_term(v) for v in self.vocab) if n
            )
            self.max_n = max(self.max_n, max((len(n.split()) for n in self._vocab), default=1))

    @classmethod
    def from_linker_dict(cls, exact_map: dict[str, str], **kw) -> "NgramExtractor":
        """Build from the same term -> CUI dictionary the linker is given."""
        return cls(vocab=list(exact_map), **kw)

    def _tokens(self, text: str) -> list[str]:
        return _TOKEN.findall(text or "")

    def extract(self, text: str) -> list[str]:
        toks = self._tokens(text)
        if not toks:
            return []
        return self._scan_vocab(toks) if self._vocab is not None else self._all_ngrams(toks)

    def _scan_vocab(self, toks: list[str]) -> list[str]:
        assert self._vocab is not None
        out: list[str] = []
        i, n = 0, len(toks)
        while i < n:
            for size in range(min(self.max_n, n - i), 0, -1):
                span = " ".join(toks[i : i + size])
                if normalize_term(span) in self._vocab:
                    out.append(span)
                    i += size
                    break
            else:
                i += 1
        return out

    def _all_ngrams(self, toks: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        n = len(toks)
        for size in range(1, self.max_n + 1):
            for i in range(n - size + 1):
                span = " ".join(toks[i : i + size])
                if size == 1 and self.drop_stopwords and span.lower() in STOPWORDS:
                    continue
                key = normalize_term(span)
                if key and key not in seen:
                    seen.add(key)
                    out.append(span)
        return out
