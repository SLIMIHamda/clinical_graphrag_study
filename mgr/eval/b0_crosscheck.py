"""B0 metric cross-check — validate baselines against published MedRAG numbers.

Step 1 go/no-go (Doc 00): No-RAG / BM25 accuracy on MIRAGE must land within
tolerance of the published MedRAG results, or the harness is suspect. This is a
correctness gate on the *harness*, not a contribution.

Reference numbers are supplied as a JSON file ({"benchmark|condition": acc}) so
they are auditable and cited to the paper — we ship an example, not hard-coded
claims. Verify the values against the MedRAG/MIRAGE paper before reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

    @property
    def passed(self) -> bool:
        return all(r.within_tol for r in self.rows) and bool(self.rows)

    def summary(self) -> str:
        lines = [f"B0 cross-check (tol +/-{self.tolerance}):"]
        for r in self.rows:
            flag = "ok " if r.within_tol else "OUT"
            lines.append(f"  [{flag}] {r.key:<28} obs={r.observed:.3f} ref={r.reference:.3f} d={r.delta:+.3f}")
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def crosscheck(
    observed: dict[str, float],
    reference: dict[str, float],
    *,
    tolerance: float = 0.05,
) -> CrossCheckReport:
    """Compare observed accuracies to reference, keyed by 'benchmark|condition'."""
    rows: list[CrossCheckRow] = []
    for key, ref in reference.items():
        if key not in observed:
            continue
        obs = observed[key]
        delta = obs - ref
        rows.append(CrossCheckRow(key, obs, ref, delta, abs(delta) <= tolerance))
    return CrossCheckReport(tolerance=tolerance, rows=rows)


def load_reference(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: float(v) for k, v in raw.items() if not k.startswith("_")}  # skip notes
