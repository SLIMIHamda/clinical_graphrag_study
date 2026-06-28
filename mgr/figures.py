"""Figure generation (Doc 3 figure plan). Headless (Agg) — writes files.

  plot_rgd            -> F3: retrieval-gain vs generation-gain per system
  plot_pareto         -> F4: CARe vs always vs never on the cost-quality frontier
  plot_coverage_curve -> F5: UMLS grounding coverage as tiers are enabled

These take the already-computed objects from the stats/metrics layers, so the
figure code carries no analysis logic (and stays testable by asserting a
non-empty file is produced).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the pod/CI
import matplotlib.pyplot as plt  # noqa: E402

from mgr.rerank.care_gate import FrontierPoint  # noqa: E402
from mgr.stats.rgd import RGDPoint  # noqa: E402


def plot_rgd(points: list[RGDPoint], out_path: str | Path, *, baseline: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in points:
        ax.scatter(p.retrieval_gain, p.generation_gain, s=60)
        ax.annotate(p.system, (p.retrieval_gain, p.generation_gain), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("retrieval gain (vs baseline)")
    ax.set_ylabel("generation gain (vs baseline)")
    ax.set_title("Retrieval–Generation Decomposition" + (f" (vs {baseline})" if baseline else ""))
    fig.tight_layout()
    return _save(fig, out_path)


def plot_pareto(frontier: Mapping[str, FrontierPoint], out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, fp in frontier.items():
        ax.scatter(fp.total_cost, fp.mean_quality, s=80)
        ax.annotate(name, (fp.total_cost, fp.mean_quality), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("rerank cost")
    ax.set_ylabel("mean quality")
    ax.set_title("Cost–Quality Frontier (CARe vs always vs never)")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_coverage_curve(curve: Mapping[str, float], out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    tiers = list(curve.keys())
    vals = [curve[t] for t in tiers]
    ax.plot(range(len(tiers)), vals, marker="o")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of mentions linked")
    ax.set_title("UMLS Grounding Coverage")
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
