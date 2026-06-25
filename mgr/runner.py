"""The runner: one manifest row -> one deterministic run (claim/execute/record).

The unit of work is a row (Doc 00 section 0). The runner never re-derives
fan-out; it resolves Ready rows against the gate ledger, claims one, executes it,
and writes a run-record. Resume skips rows whose record is present and passes
integrity checks.

Execution is pluggable via an ``Executor``. The default ``stub_executor``
exercises the full claim->record->release lifecycle deterministically *without
spending tokens* — real retrieval/generation executors bolt on in later steps
and set ``executor="real"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from mgr import ids
from mgr.metrics.cost import CostMeter
from mgr.state import Status, acquire_claim, release_claim, resolve_status
from mgr.tracking import record as rec
from manifest.manifest import Manifest, RunRow


@dataclass
class ExecResult:
    """What an executor returns for one run.

    ``expected_n_items`` is how many items the run *intended* to process: the
    full benchmark count for a real sweep, or the subset size for a smoke run.
    Integrity checks ``n_items == expected_n_items`` so a truncated data load is
    caught on full runs while a deliberate subset run is still allowed to pass.
    """

    n_items: int
    tokens_in: int = 0
    tokens_out: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    expected_n_items: int | None = None


# An Executor maps (row, effective_config) -> ExecResult.
Executor = Callable[[RunRow, dict[str, Any]], ExecResult]


def stub_executor(row: RunRow, cfg: dict[str, Any]) -> ExecResult:
    """Deterministic placeholder: 'processes' the full benchmark, spends nothing.

    Emits one per-item row per question with a stable synthetic ``qid`` so the
    downstream stats layer's qid-join can be wired and tested before any real
    generation exists.
    """
    items = [
        {
            "qid": f"{row.benchmark}_{i:05d}",
            "answer_norm": None,
            "gold": None,
            "correct": None,
            "tokens": {"in": row.tokens_per_item, "out": 0},
            "stub": True,
        }
        for i in range(row.n_items)
    ]
    return ExecResult(
        n_items=row.n_items,
        tokens_in=0,  # a stub spends no real tokens
        tokens_out=0,
        metrics={"stub": True},
        items=items,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Runner:
    manifest: Manifest
    gate_ledger: dict[str, bool]
    results_root: str | Path = "results"
    manifest_lock_hash: str | None = None
    stale_after_s: float = 3600.0

    def run_ready(self, executor: Executor = stub_executor) -> list[rec.RunRecord]:
        """Execute every Ready row (gate satisfied) that is not already Done."""
        out: list[rec.RunRecord] = []
        for row in self.manifest.resolve_ready(self.gate_ledger):
            r = self.run_one(row.run_id, executor=executor)
            if r is not None:
                out.append(r)
        return out

    def run_one(self, run_id: str, executor: Executor = stub_executor) -> rec.RunRecord | None:
        """Run a single row. Returns its record, or ``None`` if skipped.

        Skips when: already Done (resume), not Ready (gate unsatisfied), or the
        claim is held live by another worker.
        """
        row = self.manifest.by_id(run_id)

        # Resume: a present, integrity-passing Done record is never re-run.
        existing = rec.read_record(run_id, self.results_root)
        if existing is not None and rec.integrity_ok(existing, row.n_items):
            return None

        status = resolve_status(Status(row.status), row.depends_on, self.gate_ledger)
        if status != Status.READY:
            return None  # Blocked: gate not satisfied

        claim = acquire_claim(run_id, self.results_root, self.stale_after_s)
        if claim is None:
            return None  # held by a live worker

        try:
            cfg = self.manifest.effective_config(row)
            meter = CostMeter.from_estimate(row.n_items, row.tokens_per_item, row.ctx_factor, row.rate)
            started = _now()
            t0 = time.time()
            result = executor(row, cfg)
            wall_s = time.time() - t0

            items_p = None
            if result.items:
                items_p = rec.write_items(run_id, result.items, self.results_root)

            total = result.tokens_in + result.tokens_out
            meter = meter.with_actuals(total, row.rate)

            record = rec.RunRecord(
                run_id=run_id,
                slug=ids.slug(row.condition, row.benchmark, row.backbone, row.seed),
                condition=row.condition,
                benchmark=row.benchmark,
                backbone=row.backbone,
                seed=row.seed,
                status=Status.DONE.value if result.error is None else Status.FAILED.value,
                config_hash=ids.config_hash(cfg),
                manifest_lock_hash=self.manifest_lock_hash,
                code_git_sha=rec.git_sha(Path(self.results_root).resolve().parent),
                embedding_checkpoint=self.manifest.base.get("embedding_checkpoint", {}),
                n_items=result.n_items,
                started=started,
                finished=_now(),
                wall_s=wall_s,
                tokens={"in": result.tokens_in, "out": result.tokens_out, "total": total},
                cost_est_usd=meter.est_cost_usd,
                cost_actual_usd=meter.actual_cost_usd,
                metrics=result.metrics,
                items_path=items_p.as_posix() if items_p else None,
                error=result.error,
                executor="stub" if executor is stub_executor else "real",
            )

            # Done requires the run to have processed everything it intended to
            # (full benchmark for a sweep, subset size for smoke).
            expected = result.expected_n_items if result.expected_n_items is not None else result.n_items
            if record.status == Status.DONE.value and not rec.integrity_ok(record, expected):
                record.status = Status.FAILED.value
                record.error = record.error or (
                    f"integrity: processed {record.n_items} != expected {expected}"
                )

            rec.log(record, self.results_root)
            return record
        finally:
            release_claim(run_id, self.results_root)

    def rollup(self) -> Path | None:
        """Upsert all per-run records into the Parquet rollup."""
        return rec.roll_up_parquet(self.results_root)
