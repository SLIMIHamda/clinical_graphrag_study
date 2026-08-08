"""Cross-encoder reranker — the expensive step CARe gates.

At full scale this wraps a 1B reranker served on the NIM free tier (Doc 00
section 5). The interface is injectable so the CARe policy and the executor can
be tested without a model: a scorer maps (query, passage) -> relevance, and the
reranker reorders the candidate window by that score.

Cost model: reranking is O(c) forward passes per query; CARe reduces the
*expected* cost to E[g] * c by only firing on queries the gate selects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class Scorer(Protocol):
    def __call__(self, query: str, passage: str) -> float: ...


@dataclass
class CrossEncoderReranker:
    scorer: Scorer

    def rerank(self, query: str, candidate_ids: list[str], passages: dict[str, str]) -> list[str]:
        """Reorder candidates by cross-encoder relevance (desc), id-tiebroken."""
        scored = [(cid, self.scorer(query, passages.get(cid, ""))) for cid in candidate_ids]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in scored]


def constant_scorer(value: float = 0.0) -> Callable[[str, str], float]:
    """A no-op scorer (keeps input order via stable tie-break) for tests."""
    return lambda _q, _p: value


BatchScorer = Callable[[list[tuple[str, str]]], list[float]]


@dataclass
class HFCrossEncoderReranker:
    """Batched cross-encoder reranker backed by a HuggingFace sequence-classifier.

    Default ``ncbi/MedCPT-Cross-Encoder`` is the reranker MedRAG uses for
    biomedical QA — domain-matched and competitive with a general 4B model here,
    while running locally (Kaggle GPU/CPU) instead of depending on the hosted NIM
    reranker (which 404'd on the free tier). Same ``.rerank(query, ids,
    passages)`` interface as :class:`CrossEncoderReranker`, so it drops into the
    fusion retriever and the CARe oracle unchanged.

    torch/transformers are imported lazily in :meth:`load`, so importing this
    module (and everything that pulls in the fusion retriever) never requires
    them. Inject ``score_batch`` to unit-test the reorder logic without a model.
    """

    model_id: str = "ncbi/MedCPT-Cross-Encoder"
    max_length: int = 512
    batch_size: int = 32
    device: str | None = None
    score_batch: BatchScorer | None = None
    _loaded: bool = field(default=False, repr=False)

    def load(self) -> "HFCrossEncoderReranker":
        """Materialise the model. Idempotent; a no-op if ``score_batch`` is set."""
        if self.score_batch is not None or self._loaded:
            return self
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id).eval()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        def _score(pairs: list[tuple[str, str]]) -> list[float]:
            out: list[float] = []
            for i in range(0, len(pairs), self.batch_size):
                b = pairs[i : i + self.batch_size]
                enc = tok(
                    text=[p[0] for p in b],
                    text_pair=[p[1] for p in b],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                col = logits[:, 0] if logits.ndim == 2 else logits  # num_labels==1 -> (B,1)
                out.extend(float(x) for x in col.detach().cpu().tolist())
            return out

        self.score_batch = _score
        self._loaded = True
        return self

    def rerank(self, query: str, candidate_ids: list[str], passages: dict[str, str]) -> list[str]:
        """Reorder candidates by cross-encoder relevance (desc), id-tiebroken."""
        ids = list(candidate_ids)
        if not ids:
            return ids
        if self.score_batch is None:
            self.load()
        scores = self.score_batch([(query, passages.get(c, "")) for c in ids])
        order = sorted(range(len(ids)), key=lambda j: (-float(scores[j]), ids[j]))
        return [ids[j] for j in order]
