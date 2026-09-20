"""B0 metric cross-check — validate baselines against published MedRAG numbers.

Step 1 go/no-go (Doc 00): No-RAG / BM25 accuracy on MIRAGE must land within
tolerance of the published MedRAG results, or the harness is suspect. This is a
correctness gate on the *harness*, not a contribution.

Reference numbers live in a cited JSON file ({"benchmark|condition": acc}) so
they stay auditable. ``configs/b0_reference.json`` carries the MIRAGE numbers
from Xiong et al. (2024), Table 7 — the no-retrieval baseline and the
BM25-over-Textbooks row, on **GPT-3.5**.

Two checks, because they answer different questions:

* :func:`crosscheck` — absolute accuracy against the published level. Only
  meaningful when the backbone matches, which is why the report carries a
  backbone field and refuses to claim a pass across a mismatch. The paper covers
  no Llama-3.1-class model, so on our runs this is informational, not a gate.
* :func:`delta_crosscheck` — the sign and size of (condition − baseline). A
  weaker backbone scores lower everywhere, but retrieval should still *help*
  where the literature says it helps. That comparison survives a backbone
  mismatch, so it is the gate that actually applies to us.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CrossCheckRow:
    key: str
    observed: float
    reference: float
    delta: float
    within_tol: bool


@dataclass
class CrossCheckReport:
    tolerance: float
    rows: list[CrossCheckRow]
    observed_backbone: str | None = None
    reference_backbone: str | None = None

    @property
    def backbone_mismatch(self) -> bool:
        return bool(
            self.observed_backbone
            and self.reference_backbone
            and self.observed_backbone != self.reference_backbone
        )

    @property
    def passed(self) -> bool:
        if self.backbone_mismatch:
            return False  # an absolute comparison across backbones cannot pass
        return all(r.within_tol for r in self.rows) and bool(self.rows)

    def summary(self) -> str:
        lines = [f"B0 cross-check (tol +/-{self.tolerance}):"]
        if self.backbone_mismatch:
            lines.append(
                f"  [N/A] observed backbone {self.observed_backbone!r} != reference "
                f"{self.reference_backbone!r} — absolute levels are not comparable; "
                f"use delta_crosscheck"
            )
        for r in self.rows:
            flag = "ok " if r.within_tol else "OUT"
            lines.append(f"  [{flag}] {r.key:<28} obs={r.observed:.3f} ref={r.reference:.3f} d={r.delta:+.3f}")
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


@dataclass
class DeltaRow:
    benchmark: str
    condition: str
    observed_delta: float     # observed[cond] - observed[baseline]
    reference_delta: float    # reference[cond] - reference[baseline]
    sign_agrees: bool
    gap: float                # observed_delta - reference_delta


@dataclass
class DeltaReport:
    baseline: str
    rows: list[DeltaRow] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.sign_agrees for r in self.rows) and bool(self.rows)

    def summary(self) -> str:
        lines = [f"B0 delta check vs {self.baseline} (backbone-robust):"]
        for r in self.rows:
            flag = "ok " if r.sign_agrees else "OUT"
            lines.append(
                f"  [{flag}] {r.benchmark}|{r.condition:<12} obs={r.observed_delta:+.3f} "
                f"ref={r.reference_delta:+.3f} gap={r.gap:+.3f}"
            )
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def crosscheck(
    observed: dict[str, float],
    reference: dict[str, float],
    *,
    tolerance: float = 0.05,
    observed_backbone: str | None = None,
    reference_backbone: str | None = None,
) -> CrossCheckReport:
    """Compare observed accuracies to reference, keyed by 'benchmark|condition'."""
    rows: list[CrossCheckRow] = []
    for key, ref in reference.items():
        if key not in observed:
            continue
        obs = observed[key]
        delta = obs - ref
        rows.append(CrossCheckRow(key, obs, ref, delta, abs(delta) <= tolerance))
    return CrossCheckReport(
        tolerance=tolerance, rows=rows,
        observed_backbone=observed_backbone, reference_backbone=reference_backbone,
    )


def delta_crosscheck(
    observed: dict[str, float],
    reference: dict[str, float],
    *,
    baseline: str = "No-RAG",
) -> DeltaReport:
    """Does retrieval move accuracy the same *direction* as in the literature?

    A weaker backbone scores lower everywhere, so absolute levels drift. Whether
    BM25 beats closed-book on a given benchmark should not: if the published
    delta is positive and ours is negative, retrieval is broken in our harness
    (bad chunking, truncated context, a too-small corpus) rather than merely
    weaker. That is the failure the first POC hit and could not name.
    """
    report = DeltaReport(baseline=baseline)
    for key, ref in reference.items():
        benchmark, _, condition = key.partition("|")
        if condition == baseline:
            continue
        base_key = f"{benchmark}|{baseline}"
        if not all(k in observed for k in (key, base_key)) or base_key not in reference:
            continue
        obs_delta = observed[key] - observed[base_key]
        ref_delta = ref - reference[base_key]
        report.rows.append(
            DeltaRow(
                benchmark=benchmark, condition=condition,
                observed_delta=obs_delta, reference_delta=ref_delta,
                sign_agrees=(obs_delta >= 0) == (ref_delta >= 0),
                gap=obs_delta - ref_delta,
            )
        )
    report.rows.sort(key=lambda r: (r.benchmark, r.condition))
    return report


class UnverifiedReference(RuntimeError):
    """The reference file has not been filled in from the published paper."""


def load_reference(path: str | Path, *, require_verified: bool = True) -> dict[str, float]:
    """Load ``{"benchmark|condition": acc}``, refusing an unfilled file.

    The shipped reference starts with ``_verified: false`` and null values. A
    placeholder that silently loads is worse than no gate at all: it turns "we
    never checked against the literature" into a passing cross-check, and every
    downstream accuracy claim inherits that. ``require_verified=False`` is for
    tests that supply their own numbers.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if require_verified and not raw.get("_verified", False):
        raise UnverifiedReference(
            f"{path} is not verified: fill each value from the published MedRAG/MIRAGE "
            f"paper, record the citation in '_source', then set '_verified': true"
        )
    values = {k: v for k, v in raw.items() if not k.startswith("_")}  # skip notes/metadata
    missing = sorted(k for k, v in values.items() if v is None)
    if missing and require_verified:
        raise UnverifiedReference(f"{path} has unfilled entries: {missing}")
    return {k: float(v) for k, v in values.items() if v is not None}


@dataclass
class ReferenceSet:
    """Reference values plus the provenance that says what they apply to."""

    values: dict[str, float]
    backbone: str | None = None
    corpus: str | None = None
    source: str | None = None


def load_reference_set(path: str | Path, *, require_verified: bool = True) -> ReferenceSet:
    """:func:`load_reference` plus the ``_backbone`` / ``_corpus`` / ``_source`` metadata.

    The metadata is not decoration: a reference is only a harness check for the
    backbone and corpus it was measured on, and carrying that alongside the
    numbers is what lets :func:`crosscheck` refuse a meaningless pass.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReferenceSet(
        values=load_reference(path, require_verified=require_verified),
        backbone=raw.get("_backbone"),
        corpus=raw.get("_corpus"),
        source=raw.get("_source"),
    )
