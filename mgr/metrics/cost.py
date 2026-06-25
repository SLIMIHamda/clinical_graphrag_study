"""Cost meter — token side (Doc 00 section 6, meter (a)).

Two meters exist in the program:
  (a) manifest token estimate — reproduces est_tokens / est_cost, tracked
      against the $591 / $887 envelope. Implemented here.
  (b) actual = pod_uptime_h * $/h — the GPU-hours number (~$280-320 target),
      wired in the RunPod infra layer in a later step.

Because RunPod is billed by GPU-hour rather than per-token, (b) << (a) when the
A100 is saturated; both are reported so the budget reconciles either way.
"""

from __future__ import annotations

from dataclasses import dataclass


def est_tokens(n_items: int, tokens_per_item: int, ctx_factor: float) -> float:
    """Manifest cost model: n_items * tokens/item * ctx_factor."""
    return n_items * tokens_per_item * ctx_factor


def est_cost_usd(n_items: int, tokens_per_item: int, ctx_factor: float, rate_per_m: float) -> float:
    """est_cost = est_tokens / 1e6 * rate."""
    return est_tokens(n_items, tokens_per_item, ctx_factor) / 1e6 * rate_per_m


def token_cost_usd(tokens_total: int, rate_per_m: float) -> float:
    """Cost of *actually observed* tokens at the backbone's blended rate."""
    return tokens_total / 1e6 * rate_per_m


@dataclass(frozen=True)
class CostMeter:
    """Per-run token accounting: estimate (manifest) vs actual (observed)."""

    est_tokens: float
    est_cost_usd: float
    actual_tokens: int = 0
    actual_cost_usd: float = 0.0

    @classmethod
    def from_estimate(cls, n_items: int, tokens_per_item: int, ctx_factor: float, rate_per_m: float) -> "CostMeter":
        et = est_tokens(n_items, tokens_per_item, ctx_factor)
        return cls(est_tokens=et, est_cost_usd=et / 1e6 * rate_per_m)

    def with_actuals(self, tokens_total: int, rate_per_m: float) -> "CostMeter":
        return CostMeter(
            est_tokens=self.est_tokens,
            est_cost_usd=self.est_cost_usd,
            actual_tokens=tokens_total,
            actual_cost_usd=token_cost_usd(tokens_total, rate_per_m),
        )
