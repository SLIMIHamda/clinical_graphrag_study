"""CARe — Cost-Aware Adaptive Reranking gate (contribution C3).

A per-query gate g(q) in {0,1} that predicts whether cross-encoder reranking
will improve the final answer, so reranking compute is spent only on ambiguous
queries (Doc 1 section 4.3). Features are cheap and pre-generation:

  - candidate score dispersion (top-1 vs top-k fusion-score gap)
  - concept-overlap entropy across candidates
  - fraction of near-tied documents
  - a query-type signal

Training signal = the *oracle* benefit (did reranking actually help that query),
labelled from B3 + M1 results. Deployment converts "reranking sometimes helps"
into a decision rule on the cost-quality frontier: rerank iff the expected
quality gain exceeds its latency/token cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = ("top1_gap", "top1_topk_gap", "score_std", "near_tie_frac", "overlap_entropy", "query_type")


@dataclass(frozen=True)
class CareFeatures:
    top1_gap: float          # s1 - s2 over max-normalized fusion scores
    top1_topk_gap: float     # s1 - s_k
    score_std: float         # spread of the top-k scores
    near_tie_frac: float     # fraction of candidates within tol of the top
    overlap_entropy: float   # normalized entropy of concept-overlap mass
    query_type: float        # external signal (e.g. 1=multi-hop)

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in FEATURE_NAMES], dtype=float)


def _entropy(weights: list[float]) -> float:
    total = sum(w for w in weights if w > 0)
    if total <= 0:
        return 0.0
    ps = [w / total for w in weights if w > 0]
    if len(ps) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ps))  # normalized to [0, 1]


def extract_features(
    scores: list[float],
    overlaps: list[float] | None = None,
    *,
    k: int = 5,
    tie_tol: float = 0.05,
    query_type: float = 0.0,
) -> CareFeatures:
    """Build CARe features from a fused candidate list (scores desc)."""
    if not scores:
        return CareFeatures(0.0, 0.0, 0.0, 0.0, 0.0, query_type)
    s = sorted(scores, reverse=True)
    top = s[0] or 1.0
    norm = [x / top for x in s]  # max-normalize so gaps are scale-free
    topk = norm[:k]
    s1 = norm[0]
    s2 = norm[1] if len(norm) > 1 else 0.0
    sk = topk[-1]
    near_tie = sum(1 for x in norm if (s1 - x) <= tie_tol) / len(norm)
    return CareFeatures(
        top1_gap=s1 - s2,
        top1_topk_gap=s1 - sk,
        score_std=float(np.std(topk)) if len(topk) > 1 else 0.0,
        near_tie_frac=near_tie,
        overlap_entropy=_entropy(overlaps) if overlaps else 0.0,
        query_type=query_type,
    )


def oracle_benefit(quality_with_rerank: float, quality_without: float, *, eps: float = 0.0) -> int:
    """Oracle label: 1 iff reranking strictly improved answer quality."""
    return int(quality_with_rerank - quality_without > eps)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class CareGate:
    """Logistic gate over CARe features, with a cost-aware decision rule."""

    weights: np.ndarray
    bias: float
    mu: np.ndarray            # feature standardization (fit-time)
    sigma: np.ndarray
    threshold: float = 0.5

    def predict_proba(self, f: CareFeatures) -> float:
        x = (f.as_vector() - self.mu) / self.sigma
        return float(_sigmoid(np.array([x @ self.weights + self.bias]))[0])

    def decide(self, f: CareFeatures, *, value: float = 1.0, cost: float = 0.0) -> bool:
        """Rerank iff expected gain beats cost.

        With cost=0 this is a plain probability threshold. With a positive cost,
        rerank iff ``p * value > cost`` (expected quality gain exceeds spend).
        """
        p = self.predict_proba(f)
        if cost > 0.0:
            return p * value > cost
        return p >= self.threshold

    @classmethod
    def fit(
        cls,
        features: list[CareFeatures],
        labels: list[int],
        *,
        lr: float = 0.1,
        epochs: int = 2000,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> "CareGate":
        """Train the gate on (features, oracle_label) by logistic regression."""
        X = np.array([f.as_vector() for f in features], dtype=float)
        y = np.array(labels, dtype=float)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma == 0] = 1.0
        Xs = (X - mu) / sigma
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.01, size=Xs.shape[1])
        b = 0.0
        n = len(y)
        for _ in range(epochs):
            p = _sigmoid(Xs @ w + b)
            grad_w = Xs.T @ (p - y) / n + l2 * w
            grad_b = float(np.sum(p - y) / n)
            w -= lr * grad_w
            b -= lr * grad_b
        return cls(weights=w, bias=b, mu=mu, sigma=sigma)


@dataclass
class FrontierPoint:
    policy: str
    rerank_rate: float       # fraction of queries reranked
    mean_quality: float
    total_cost: float


def cost_quality_frontier(
    decisions: list[bool],
    quality_with_rerank: list[float],
    quality_without: list[float],
    *,
    rerank_cost: float = 1.0,
) -> dict[str, FrontierPoint]:
    """Compare CARe vs always- vs never-rerank on quality and cost.

    Quality per query = with-rerank value when that policy reranks it, else the
    no-rerank value. Cost = rerank_cost per reranked query (CARe pays E[g]*c).
    """
    n = len(decisions)
    if not (n == len(quality_with_rerank) == len(quality_without)):
        raise ValueError("decisions and quality arrays must align")

    def evaluate(name: str, gates: list[bool]) -> FrontierPoint:
        q = [quality_with_rerank[i] if gates[i] else quality_without[i] for i in range(n)]
        rate = sum(gates) / n if n else 0.0
        return FrontierPoint(name, rate, float(np.mean(q)) if q else 0.0, sum(gates) * rerank_cost)

    return {
        "care": evaluate("care", decisions),
        "always": evaluate("always", [True] * n),
        "never": evaluate("never", [False] * n),
    }
