"""Effect sizes reported alongside every p (Doc 00 section 4, step 5).

Paired Cohen's d for magnitude on the difference scale, and Cliff's delta for a
non-parametric dominance measure that is robust to the binary/ordinal metrics
(EM, correct) common here.
"""

from __future__ import annotations

import numpy as np


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d: mean(d) / sd(d). Returns 0.0 when there is no variance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(d.mean() / sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]: P(a>b) - P(a<b) over all pairs.

    +1 means every a exceeds every b; 0 means stochastic equality.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("cliffs_delta needs non-empty samples")
    diff = a[:, None] - b[None, :]
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    return float((gt - lt) / (len(a) * len(b)))


def interpret_cliffs(delta: float) -> str:
    """Romano et al. thresholds for |delta|."""
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"
