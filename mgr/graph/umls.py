"""UMLS concept grounding — exact + abbreviation + fuzzy linking, with coverage.

Grounds entity mentions to UMLS CUIs. The thesis linked only ~47% of entity
nodes by exact match; this adds abbreviation expansion and fuzzy matching and
reports coverage as a *measured curve* (Doc 00 control #7 / Doc 1 section 6).

The grounded concepts feed CA-RRF (a candidate's concept set, the query's
concept set) and the coverage figure (F5). The linker operates on pre-extracted
mentions; the NER/term-extraction step (scispaCy/QuickUMLS at scale) is injected
upstream, so the linker's correctness is independent of corpus size.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping


def normalize_term(term: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", term.lower())).strip()


@dataclass
class LinkResult:
    term: str
    cui: str | None
    method: str  # exact | abbrev | fuzzy | none


@dataclass
class UMLSLinker:
    """Links mentions to CUIs via exact, then abbreviation, then fuzzy matching."""

    exact_map: Mapping[str, str]            # normalized surface form -> CUI
    abbrev_map: Mapping[str, str] | None = None  # abbreviation -> CUI (or expansion)
    fuzzy_threshold: float = 0.88
    fuzzy_enabled: bool = True

    def __post_init__(self) -> None:
        self._exact = {normalize_term(k): v for k, v in self.exact_map.items()}
        self._abbrev = {normalize_term(k): v for k, v in (self.abbrev_map or {}).items()}
        self._keys = list(self._exact.keys())

    def link_term(self, term: str) -> LinkResult:
        t = normalize_term(term)
        if not t:
            return LinkResult(term, None, "none")
        if t in self._exact:
            return LinkResult(term, self._exact[t], "exact")
        if t in self._abbrev:
            cui = self._abbrev[t]
            # an abbreviation may map to an expansion that is itself in the dict
            return LinkResult(term, self._exact.get(normalize_term(cui), cui), "abbrev")
        if self.fuzzy_enabled and self._keys:
            match = difflib.get_close_matches(t, self._keys, n=1, cutoff=self.fuzzy_threshold)
            if match:
                return LinkResult(term, self._exact[match[0]], "fuzzy")
        return LinkResult(term, None, "none")

    def link_mentions(self, terms: Iterable[str]) -> list[LinkResult]:
        return [self.link_term(t) for t in terms]

    def concepts(self, terms: Iterable[str]) -> set[str]:
        """The set of CUIs grounded from a list of mentions (linked only)."""
        return {r.cui for r in self.link_mentions(terms) if r.cui is not None}


@dataclass
class CoverageReport:
    total: int
    exact: int
    abbrev: int
    fuzzy: int

    @property
    def unlinked(self) -> int:
        return self.total - self.exact - self.abbrev - self.fuzzy

    @property
    def curve(self) -> dict[str, float]:
        """Cumulative coverage as each linking tier is enabled (the F5 curve)."""
        if self.total == 0:
            return {"exact": 0.0, "exact+abbrev": 0.0, "exact+abbrev+fuzzy": 0.0}
        e, a, f = self.exact, self.abbrev, self.fuzzy
        return {
            "exact": e / self.total,
            "exact+abbrev": (e + a) / self.total,
            "exact+abbrev+fuzzy": (e + a + f) / self.total,
        }


def coverage(terms: Iterable[str], linker: UMLSLinker) -> CoverageReport:
    """Measure linking coverage by method over a set of mentions."""
    counts = {"exact": 0, "abbrev": 0, "fuzzy": 0, "none": 0}
    total = 0
    for r in linker.link_mentions(terms):
        counts[r.method] += 1
        total += 1
    return CoverageReport(total=total, exact=counts["exact"], abbrev=counts["abbrev"], fuzzy=counts["fuzzy"])
