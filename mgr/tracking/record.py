"""Canonical experiment record: per-run JSON + per-item JSONL, rolled up to Parquet.

Canonical = JSONL/Parquet + DuckDB (Doc 00 section 4). Dashboards are *views*,
never the record of truth. Write path:

    record.log(run_record)  ->  append results/per-run/{run_id}.json
                                upsert results/run_records.parquet

The per-item JSONL is the substrate the stats layer joins on by ``qid``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mgr import ids


def git_sha(cwd: str | Path | None = None) -> str | None:
    """Best-effort short git sha of the code; ``None`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class RunRecord:
    """Run-record schema (Doc 00 section 3.2)."""

    run_id: str
    slug: str
    condition: str
    benchmark: str
    backbone: str
    seed: int
    status: str
    config_hash: str
    manifest_lock_hash: str | None = None
    code_git_sha: str | None = None
    embedding_checkpoint: dict[str, Any] = field(default_factory=dict)
    prompt_set_hash: str | None = None
    graph_hash: str | None = None
    index_hashes: dict[str, Any] = field(default_factory=dict)
    n_items: int = 0
    started: str | None = None
    finished: str | None = None
    wall_s: float = 0.0
    tokens: dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0, "total": 0})
    cost_est_usd: float = 0.0
    cost_actual_usd: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    items_path: str | None = None
    runpod_pod_id: str | None = None
    env_lock_hash: str | None = None
    error: str | None = None
    executor: str = "stub"  # which executor produced this record (stub | real)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log(record: RunRecord, results_root: str | Path = "results") -> Path:
    """Write a run-record to its canonical JSON path."""
    path = ids.record_path(record.run_id, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_record(run_id: str, results_root: str | Path = "results") -> RunRecord | None:
    path = ids.record_path(run_id, results_root)
    if not path.exists():
        return None
    return RunRecord(**json.loads(path.read_text(encoding="utf-8")))


def write_items(run_id: str, items: Iterable[dict[str, Any]], results_root: str | Path = "results") -> Path:
    """Append per-item rows to results/per-run/{run_id}/items.jsonl."""
    path = ids.items_path(run_id, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def integrity_ok(record: RunRecord, expected_n_items: int) -> bool:
    """Done iff the run processed the full benchmark and emitted metrics.

    (Doc 00 section 3.1: n_items processed == benchmark n_items, metrics emitted.)
    """
    return (
        record.status == "Done"
        and record.n_items == expected_n_items
        and bool(record.metrics)
        and record.error is None
    )


def roll_up_parquet(results_root: str | Path = "results") -> Path | None:
    """Upsert all per-run JSON records into results/run_records.parquet.

    Returns the parquet path, or ``None`` if pyarrow is unavailable (the JSON
    records remain the canonical truth regardless).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None

    root = Path(results_root)
    per_run = root / "per-run"
    if not per_run.exists():
        return None
    records = []
    for jf in sorted(per_run.glob("R*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        # Flatten nested dicts to JSON strings for a stable columnar schema.
        for k in ("embedding_checkpoint", "index_hashes", "tokens", "metrics"):
            if isinstance(d.get(k), (dict, list)):
                d[k] = json.dumps(d[k], ensure_ascii=False)
        records.append(d)
    if not records:
        return None
    out = root / "run_records.parquet"
    pq.write_table(pa.Table.from_pylist(records), out)
    return out
