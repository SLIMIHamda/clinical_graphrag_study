"""FusionRetriever — the hybrid arm that ties the contributions together.

One retriever that: runs each component retriever (lexical/dense/graph), fuses
their ranked lists with CA-RRF (concept list on/off = the grounding ablation),
and optionally reranks the top window — never (off), always (static), or only
when the CARe gate fires (adaptive). The single class expresses every hybrid
condition by flipping ``use_concept`` and ``rerank_mode``:

  Hybrid-RRF2/RRF4            -> use_concept=False, rerank=off
  Hybrid-CARRF               -> use_concept=True,  rerank=off
  Hybrid-CARRF-staticRerank  -> use_concept=True,  rerank=static
  Hybrid-CARRF-CARe          -> use_concept=True,  rerank=adaptive   (HEADLINE)
  Hybrid-CARRF-noGrounding   -> use_concept=False, rerank=off
  Hybrid-CARRF-noVecIndex    -> as CARRF, but the dense component uses a flat index

The executor and prompt path are unchanged across all of these (prompt parity);
the condition is the only manipulated factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .base import RetrievalResult, Retriever, render_context
from .ca_rrf import ca_rrf, concept_overlap_scores
from mgr.rerank.care_gate import CareGate, extract_features
from mgr.rerank.cross_encoder import CrossEncoderReranker

# condition -> (use_concept, rerank_mode)
HYBRID_SPECS: dict[str, tuple[bool, str]] = {
    "Hybrid-RRF2": (False, "off"),
    "Hybrid-RRF4": (False, "off"),
    "Hybrid-CARRF": (True, "off"),
    "Hybrid-CARRF-staticRerank": (True, "static"),
    "Hybrid-CARRF-CARe": (True, "adaptive"),
    "Hybrid-CARRF-noVecIndex": (True, "off"),
    "Hybrid-CARRF-noGrounding": (False, "off"),
}


@dataclass
class FusionRetriever:
    components: dict[str, Retriever]
    passages: dict[str, str]
    use_concept: bool = False
    rerank_mode: str = "off"  # off | static | adaptive
    query_concepts_fn: Callable[[str], set[str]] | None = None
    candidate_concepts: Mapping[str, set[str]] | None = None
    care_gate: CareGate | None = None
    reranker: CrossEncoderReranker | None = None
    k: int = 60
    weights: dict[str, float] | None = None
    concept_metric: str = "count"
    rerank_window: int = 10
    query_type: float = 0.0
    max_context_chars: int = 8000

    def __post_init__(self) -> None:
        if self.use_concept and self.query_concepts_fn is None:
            raise ValueError("use_concept requires query_concepts_fn")
        if self.rerank_mode not in ("off", "static", "adaptive"):
            raise ValueError(f"bad rerank_mode: {self.rerank_mode!r}")
        if self.rerank_mode != "off" and self.reranker is None:
            raise ValueError(f"rerank_mode={self.rerank_mode} requires a reranker")
        if self.rerank_mode == "adaptive" and self.care_gate is None:
            raise ValueError("adaptive rerank requires a CARe gate")

    def retrieve(self, query: str, *, depth_k: int = 10) -> RetrievalResult:
        pool = max(depth_k, self.rerank_window, 20)
        comp_lists = {
            name: r.retrieve(query, depth_k=pool).retrieved_ids for name, r in self.components.items()
        }
        qc = set(self.query_concepts_fn(query)) if (self.use_concept and self.query_concepts_fn) else set()
        cc = self.candidate_concepts or {}

        fused = ca_rrf(
            comp_lists, qc, cc, k=self.k, weights=self.weights,
            concept_metric=self.concept_metric, use_concept=self.use_concept,
        )
        fused_ids = [d for d, _ in fused]
        score_map = dict(fused)

        rerank_fired = self._maybe_rerank(query, fused_ids, score_map, qc, cc)
        if rerank_fired:
            window = self.reranker.rerank(query, fused_ids[: self.rerank_window], self.passages)
            fused_ids = window + fused_ids[self.rerank_window :]

        final = fused_ids[:depth_k]
        return RetrievalResult(
            context=render_context(final, self.passages, self.max_context_chars),
            retrieved_ids=final,
            ranks={d: i + 1 for i, d in enumerate(final)},
            fused_scores=[score_map.get(d, 0.0) for d in final],
            rerank_fired=rerank_fired,
        )

    def _maybe_rerank(self, query, fused_ids, score_map, qc, cc) -> bool:
        if self.rerank_mode == "off" or not fused_ids:
            return False
        if self.rerank_mode == "static":
            return True
        # adaptive: decide from CARe features over the candidate pool
        ids = fused_ids[: self.rerank_window]
        scores = [score_map[d] for d in ids]
        overlaps = None
        if self.use_concept and qc:
            ov = concept_overlap_scores(qc, {d: set(cc.get(d, set())) for d in ids}, metric=self.concept_metric)
            overlaps = [ov[d] for d in ids]
        feats = extract_features(scores, overlaps, query_type=self.query_type)
        return bool(self.care_gate.decide(feats))


def build_fusion(
    condition: str,
    *,
    components: dict[str, Retriever],
    passages: dict[str, str],
    query_concepts_fn: Callable[[str], set[str]] | None = None,
    candidate_concepts: Mapping[str, set[str]] | None = None,
    care_gate: CareGate | None = None,
    reranker: CrossEncoderReranker | None = None,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> FusionRetriever:
    """Construct the FusionRetriever for a hybrid condition from its spec."""
    if condition not in HYBRID_SPECS:
        raise KeyError(f"{condition!r} is not a hybrid condition; specs: {sorted(HYBRID_SPECS)}")
    use_concept, rerank_mode = HYBRID_SPECS[condition]
    return FusionRetriever(
        components=components,
        passages=passages,
        use_concept=use_concept,
        rerank_mode=rerank_mode,
        query_concepts_fn=query_concepts_fn,
        candidate_concepts=candidate_concepts,
        care_gate=care_gate,
        reranker=reranker,
        k=k,
        weights=weights,
    )
