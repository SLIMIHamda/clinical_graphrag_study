"""Holm-Bonferroni correction across a comparison family (Doc 00 section 4, step 4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HolmResult:
    label: str
    p_raw: float
    p_adj: float
    reject: bool


def holm_bonferroni(pvalues: dict[str, float], *, alpha: float = 0.05) -> list[HolmResult]:
    """Step-down Holm-Bonferroni. Returns results in ascending raw-p order.

    Adjusted p_i = max over the step-down prefix of (m - rank) * p, clamped to
    <= 1 and made monotone non-decreasing, which is the standard reporting form.
    """
    m = len(pvalues)
    if m == 0:
        return []
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    results: list[HolmResult] = []
    prev_adj = 0.0
    still_rejecting = True
    for i, (label, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev_adj)  # enforce monotonicity
        prev_adj = adj
        # Holm rejects p_i while every earlier (smaller) p also passed its threshold.
        if still_rejecting and p <= alpha / (m - i):
            reject = True
        else:
            reject = False
            still_rejecting = False
        results.append(HolmResult(label=label, p_raw=p, p_adj=adj, reject=reject))
    return results
