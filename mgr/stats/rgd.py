"""Retrieval-Generation Decomposition (RGD) — contribution C1.

The headline empirical object: for each system, the gain in *retrieval* quality
and the gain in *generation* quality versus a baseline, measured on the same
items. The thesis symptom is that graph systems top retrieval (Recall@3/MRR)
while losing answer quality; RGD formalizes this as a measured gap and flags
where the two diverge (Doc 1 section 4.1).

This computes the per-system (retrieval_gain, generation_gain) points that the
decomposition figure F3 plots; the per-query-type regression lives alongside the
stats layer once item-level type tags are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RGDPoint:
    system: str
    retrieval: float
    generation: float
    retrieval_gain: float
    generation_gain: float

    @property
    def diverges(self) -> bool:
        """Retrieval improves but generation does not (the C1 phenomenon)."""
        return self.retrieval_gain > 0.0 and self.generation_gain <= 0.0


def decompose(
    systems: Mapping[str, tuple[float, float]],
    baseline: str,
) -> list[RGDPoint]:
    """Decompose systems into retrieval/generation gains vs a baseline.

    ``systems`` maps name -> (retrieval_metric, generation_metric) on the same
    items (e.g. Recall@3 and answer accuracy). Returns points sorted by
    retrieval gain, descending; the baseline itself is included with zero gains.
    """
    if baseline not in systems:
        raise KeyError(f"baseline {baseline!r} not among systems")
    br, bg = systems[baseline]
    points = [
        RGDPoint(
            system=name,
            retrieval=r,
            generation=g,
            retrieval_gain=r - br,
            generation_gain=g - bg,
        )
        for name, (r, g) in systems.items()
    ]
    points.sort(key=lambda p: p.retrieval_gain, reverse=True)
    return points


def divergent_systems(points: list[RGDPoint]) -> list[str]:
    """Systems where graph-style selection gains don't transfer to answers."""
    return [p.system for p in points if p.diverges]
