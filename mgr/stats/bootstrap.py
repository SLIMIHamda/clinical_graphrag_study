"""Bootstrap confidence intervals (Doc 00 section 4, step 2).

10k resamples -> 95% CI per primary metric. Reported on every headline number
(Doc 1 section 6). Percentile method; deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n_boot: int
    level: float


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(point=float(values.mean()), lo=float(lo), hi=float(hi), n_boot=n_boot, level=level)


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> CI:
    """Paired bootstrap CI for mean(a - b) (same items, resampled together)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length samples")
    d = a - b
    ci = bootstrap_ci(d, n_boot=n_boot, level=level, seed=seed)
    return ci
