"""Concept-Aware Reciprocal Rank Fusion (CA-RRF) — contribution C2.

CA-RRF treats UMLS-concept overlap as a first-class ranking signal: it forms an
extra ranked list ordered by the overlap between the query's grounded concepts
and each candidate's concept set, then RRF-fuses it with the lexical/dense (and
any other) lists using the *same* constant k (Doc 1 section 4.2).

The contribution must be isolable: ``ca_rrf(..., use_concept=False)`` is exactly
plain RRF over the same component lists with everything else frozen, so the
ablation {lexical+dense RRF} vs {+concept list} measures only the concept list's
marginal value (Doc 1: reviewers must not read it as a free parameter).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .rrf import reciprocal_rank_fusion


def concept_overlap_scores(
    query_concepts: set[str],
    candidate_concepts: Mapping[str, set[str]],
    *,
    metric: str = "count",
) -> dict[str, float]:
    """Overlap between query concepts and each candidate's concept set.

    ``count``  -> size of the intersection (terminology hits).
    ``jaccard``-> |intersection| / |union| (length-normalized).
    """
    q = set(query_concepts)
    out: dict[str, float] = {}
    for doc, concepts in candidate_concepts.items():
        c = set(concepts)
        inter = len(q & c)
        if metric == "jaccard":
            union = len(q | c)
            out[doc] = (inter / union) if union else 0.0
        elif metric == "count":
            out[doc] = float(inter)
        else:
            raise ValueError(f"unknown overlap metric: {metric!r}")
    return out


def concept_ranked_list(
    query_concepts: set[str],
    candidate_concepts: Mapping[str, set[str]],
    *,
    metric: str = "count",
    drop_zero: bool = True,
) -> list[str]:
    """Rank candidates by concept overlap (desc), id-tiebroken.

    With ``drop_zero`` (default), candidates with no overlap are excluded so the
    concept list only *boosts* terminology-matching docs rather than re-ranking
    everything — matching the intent that it rescues concept-heavy queries.
    """
    scores = concept_overlap_scores(query_concepts, candidate_concepts, metric=metric)
    items = [(d, s) for d, s in scores.items() if (s > 0.0 or not drop_zero)]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in items]


def ca_rrf(
    component_lists: Mapping[str, Sequence[str]],
    query_concepts: set[str],
    candidate_concepts: Mapping[str, set[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    concept_metric: str = "count",
    use_concept: bool = True,
) -> list[tuple[str, float]]:
    """Fuse named component lists, optionally adding the concept-overlap list.

    ``component_lists`` e.g. {"lexical": [...], "dense": [...]}. ``use_concept``
    False reproduces plain RRF over the same components (the ablation baseline).
    """
    weights = dict(weights or {})
    names = list(component_lists.keys())
    lists: list[Sequence[str]] = [component_lists[n] for n in names]
    ws: list[float] = [weights.get(n, 1.0) for n in names]

    if use_concept:
        clist = concept_ranked_list(query_concepts, candidate_concepts, metric=concept_metric)
        lists.append(clist)
        ws.append(weights.get("concept", 1.0))

    return reciprocal_rank_fusion(lists, k=k, weights=ws)
