"""Paired permutation test, item-level, with exact p (Doc 00 section 4, step 3).

>=100k permutations; exact p; **refuses to report a resolution floor**. The
thesis reported an identical p = 0.00025 across three comparisons — exactly the
1/N floor of a too-small permutation count. Here, when the observed statistic is
never exceeded in the sampled permutations, we flag ``at_floor`` and report
``p < 1/(N+1)`` rather than a fake equality; the pipeline must then raise the
permutation count or report the bound honestly (never the floor as a point p).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PermResult:
    diff: float            # observed mean(a - b)
    p_value: float
    n_perm: int
    exact: bool
    at_floor: bool         # True iff the observed stat was never exceeded (p is a bound)

    def assert_resolved(self) -> "PermResult":
        """Refuse a resolution-floor p: raise so the caller adds permutations."""
        if self.at_floor:
            raise ValueError(
                f"permutation p hit the resolution floor (< {1 / (self.n_perm + 1):.2e}); "
                "increase n_perm or report the bound, never the floor as a point estimate"
            )
        return self


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 100_000,
    seed: int = 0,
    tol: float = 1e-12,
) -> PermResult:
    """Two-sided paired sign-flip permutation test on mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired test requires equal-length samples")
    d = a - b
    n = len(d)
    obs = float(d.mean())
    abs_obs = abs(obs)

    # Exact enumeration when the full sign-flip space is no larger than n_perm.
    if n <= 20 and (2**n) <= n_perm:
        signs = _all_sign_vectors(n)
        stats = (signs * d).mean(axis=1)
        count = int(np.sum(np.abs(stats) >= abs_obs - tol))
        total = signs.shape[0]
        return PermResult(diff=obs, p_value=count / total, n_perm=total, exact=True, at_floor=False)

    rng = np.random.default_rng(seed)
    exceed = 0
    # Stream in blocks to bound memory for large n_perm * n.
    block = max(1, min(n_perm, 2_000_000 // max(n, 1)))
    done = 0
    while done < n_perm:
        m = min(block, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        stats = (signs * d).mean(axis=1)
        exceed += int(np.sum(np.abs(stats) >= abs_obs - tol))
        done += m
    at_floor = exceed == 0
    p = (exceed + 1) / (n_perm + 1)  # add-one smoothing; never exactly 0
    return PermResult(diff=obs, p_value=p, n_perm=n_perm, exact=False, at_floor=at_floor)


def _all_sign_vectors(n: int) -> np.ndarray:
    grid = np.array(np.meshgrid(*[[-1.0, 1.0]] * n, indexing="ij"))
    return grid.reshape(n, -1).T
