"""Load + validate the Experiment-Matrix workbook; resolve runnable rows.

The workbook *is* the configuration database (Doc 00 section 2). This module is
the read+validate layer over it:

  - load the Run_Manifest, Conditions, Benchmarks, Backbones, Metrics sheets
  - validate the design invariants (sequential run_id, known gates, ...)
  - merge a row's effective config = base (+) condition (+) benchmark (+) backbone (+) seed
  - reproduce the cost model and reconcile against the sheet
  - resolve Ready rows against a gate ledger

The xlsx is the *human* source of truth for what to run. lock.py freezes a
derived, hashed machine contract (manifest.lock.json) that the runner executes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import openpyxl
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = REPO_ROOT / "manifest" / "Experiment-Matrix.xlsx"
DEFAULT_GATES = REPO_ROOT / "configs" / "gates.yaml"
DEFAULT_BASE = REPO_ROOT / "configs" / "base.yaml"

KNOWN_GATES = {"H2", "G3", "P3"}


def _sheet_records(ws) -> list[dict[str, Any]]:
    """Return a worksheet's rows as dicts keyed by the header row."""
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    out: list[dict[str, Any]] = []
    for r in rows[1:]:
        if all(c is None for c in r):
            continue
        out.append({header[i]: r[i] for i in range(len(header))})
    return out


@dataclass(frozen=True)
class RunRow:
    """One row of Run_Manifest = (condition x benchmark x backbone x seed)."""

    run_id: str
    priority: str
    phase: str
    track: str
    condition: str
    backbone: str
    benchmark: str
    seed: int
    status: str
    fusion: str | None
    rerank: str | None
    grounding: str | None
    retr_depth_k: Any
    ctx_factor: float
    n_items: int
    tokens_per_item: int
    est_tokens: float
    rate: float
    est_cost: float
    depends_on: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def gate(self) -> str:
        return str(self.depends_on).split()[0].strip()

    def computed_est_tokens(self) -> float:
        """Reproduce the manifest cost model: n_items * tokens/item * ctx_factor."""
        return self.n_items * self.tokens_per_item * self.ctx_factor

    def computed_est_cost(self) -> float:
        """est_cost = est_tokens / 1e6 * rate."""
        return self.computed_est_tokens() / 1e6 * self.rate


@dataclass
class Manifest:
    runs: list[RunRow]
    conditions: dict[str, dict[str, Any]]
    benchmarks: dict[str, dict[str, Any]]
    backbones: dict[str, dict[str, Any]]
    metrics: list[dict[str, Any]]
    base: dict[str, Any]

    def __iter__(self) -> Iterator[RunRow]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def by_id(self, run_id: str) -> RunRow:
        for r in self.runs:
            if r.run_id == run_id:
                return r
        raise KeyError(run_id)

    # ----- effective config merge ----------------------------------------- #
    def effective_config(self, run: RunRow) -> dict[str, Any]:
        """base (+) condition (+) benchmark (+) backbone (+) {seed}.

        Later layers override earlier ones on key collision. This is the dict
        that gets canonicalized + hashed into ``config_hash``.
        """
        merged: dict[str, Any] = {"base": self.base}
        merged["condition"] = self.conditions[run.condition]
        merged["benchmark"] = self.benchmarks[run.benchmark]
        merged["backbone"] = self.backbones[run.backbone]
        merged["seed"] = run.seed
        merged["run_id"] = run.run_id
        return merged

    # ----- readiness ------------------------------------------------------- #
    def resolve_ready(self, gate_ledger: dict[str, bool]) -> list[RunRow]:
        """Return rows whose gate is satisfied (Pending/Blocked -> Ready)."""
        from mgr.state import Status, resolve_status

        ready = []
        for r in self.runs:
            if resolve_status(Status(r.status), r.depends_on, gate_ledger) == Status.READY:
                ready.append(r)
        return ready


def load_gate_ledger(path: str | Path = DEFAULT_GATES) -> dict[str, bool]:
    """Load the gate ledger as ``{gate_key: satisfied_bool}``."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: bool(v.get("satisfied", False)) for k, v in doc["gates"].items()}


def _as_int(v: Any) -> int:
    return int(round(float(v)))


def load_manifest(
    xlsx: str | Path = DEFAULT_XLSX,
    base: str | Path = DEFAULT_BASE,
) -> Manifest:
    wb = openpyxl.load_workbook(xlsx, data_only=True)

    conditions = {r["condition"]: r for r in _sheet_records(wb["Conditions"]) if r.get("condition")}
    benchmarks = {r["benchmark"]: r for r in _sheet_records(wb["Benchmarks"]) if r.get("benchmark") and r.get("n_items")}
    backbones = {r["backbone"]: r for r in _sheet_records(wb["Backbones"]) if r.get("backbone")}
    metrics = _sheet_records(wb["Metrics"])

    runs: list[RunRow] = []
    for rec in _sheet_records(wb["Run_Manifest"]):
        if not rec.get("run_id"):
            continue
        runs.append(
            RunRow(
                run_id=str(rec["run_id"]),
                priority=str(rec.get("priority", "")),
                phase=str(rec.get("phase", "")),
                track=str(rec.get("track", "")),
                condition=str(rec["condition"]),
                backbone=str(rec["backbone"]),
                benchmark=str(rec["benchmark"]),
                seed=_as_int(rec["seed"]),
                status=str(rec.get("status", "Pending")),
                fusion=(None if rec.get("fusion") in (None, "—", "�") else str(rec.get("fusion"))),
                rerank=(None if rec.get("rerank") is None else str(rec.get("rerank"))),
                grounding=(None if rec.get("grounding") is None else str(rec.get("grounding"))),
                retr_depth_k=rec.get("retr_depth_k"),
                ctx_factor=float(rec["ctx_factor"]),
                n_items=_as_int(rec["n_items"]),
                tokens_per_item=_as_int(rec["tokens/item"]),
                est_tokens=float(rec["est_tokens"]),
                rate=float(rec["rate $/M"]),
                est_cost=float(rec["est_cost $"]),
                depends_on=str(rec["depends_on"]),
                raw=rec,
            )
        )

    base_cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    return Manifest(runs, conditions, benchmarks, backbones, metrics, base_cfg)


def validate(m: Manifest) -> list[str]:
    """Check the design invariants. Returns a list of human-readable problems."""
    problems: list[str] = []

    # 1. run_id is sequential R0001.. with no gaps or dups.
    from mgr.ids import format_run_id, run_id_index

    idxs = []
    for r in m.runs:
        try:
            idxs.append(run_id_index(r.run_id))
        except ValueError as e:
            problems.append(str(e))
    if idxs:
        expected = list(range(1, len(idxs) + 1))
        if sorted(idxs) != expected:
            missing = set(expected) - set(idxs)
            dups = {i for i in idxs if idxs.count(i) > 1}
            if missing:
                problems.append(f"run_id gaps: missing {sorted(format_run_id(i) for i in missing)}")
            if dups:
                problems.append(f"run_id duplicates: {sorted(format_run_id(i) for i in dups)}")

    # 2. every depends_on maps to a known gate key.
    for r in m.runs:
        if r.gate not in KNOWN_GATES:
            problems.append(f"{r.run_id}: unknown gate {r.gate!r} (depends_on={r.depends_on!r})")

    # 3. every dimension referenced by a row exists in its sheet.
    for r in m.runs:
        if r.condition not in m.conditions:
            problems.append(f"{r.run_id}: condition {r.condition!r} not in Conditions sheet")
        if r.benchmark not in m.benchmarks:
            problems.append(f"{r.run_id}: benchmark {r.benchmark!r} not in Benchmarks sheet")
        if r.backbone not in m.backbones:
            problems.append(f"{r.run_id}: backbone {r.backbone!r} not in Backbones sheet")

    # 4. cost model reconciles with the sheet (the budget meter must match).
    for r in m.runs:
        if not math.isclose(r.computed_est_tokens(), r.est_tokens, rel_tol=1e-6, abs_tol=1.0):
            problems.append(
                f"{r.run_id}: est_tokens mismatch sheet={r.est_tokens} computed={r.computed_est_tokens()}"
            )
        if not math.isclose(r.computed_est_cost(), r.est_cost, rel_tol=1e-6, abs_tol=1e-6):
            problems.append(
                f"{r.run_id}: est_cost mismatch sheet={r.est_cost} computed={r.computed_est_cost()}"
            )

    return problems


def budget_summary(m: Manifest) -> dict[str, float]:
    """Reproduce the workbook's Budget_Summary headline figures."""
    total_tokens = sum(r.computed_est_tokens() for r in m.runs)
    total_cost = sum(r.computed_est_cost() for r in m.runs)
    return {
        "n_runs": float(len(m.runs)),
        "est_tokens": total_tokens,
        "est_cost_usd": total_cost,
        "est_cost_usd_1.5x": total_cost * 1.5,
    }
