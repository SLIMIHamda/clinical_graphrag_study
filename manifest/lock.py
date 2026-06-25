"""Freeze the xlsx into a derived, hashed machine contract: manifest.lock.json.

The xlsx is the human source of truth (what to run). The lock is what the
*runner* reads: every row resolved to its effective config + config_hash, plus a
top-level manifest_lock_hash over the whole frozen set. Frozen per sweep so a
mid-sweep edit to the workbook cannot silently change a running experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mgr import ids

from .manifest import DEFAULT_XLSX, Manifest, load_manifest, validate

LOCK_PATH = Path(__file__).resolve().parent / "manifest.lock.json"


def build_lock(m: Manifest) -> dict[str, Any]:
    """Build the lock document from a loaded manifest."""
    entries = []
    for r in m.runs:
        cfg = m.effective_config(r)
        entries.append(
            {
                "run_id": r.run_id,
                "slug": ids.slug(r.condition, r.benchmark, r.backbone, r.seed),
                "condition": r.condition,
                "benchmark": r.benchmark,
                "backbone": r.backbone,
                "seed": r.seed,
                "depends_on_gate": r.gate,
                "ctx_factor": r.ctx_factor,
                "n_items": r.n_items,
                "tokens_per_item": r.tokens_per_item,
                "est_tokens": r.computed_est_tokens(),
                "rate": r.rate,
                "est_cost_usd": r.computed_est_cost(),
                "config_hash": ids.config_hash(cfg),
            }
        )
    body = {"version": 1, "n_runs": len(entries), "runs": entries}
    body["manifest_lock_hash"] = _hash_body(body)
    return body


def _hash_body(body: dict[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "manifest_lock_hash"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_lock(m: Manifest, path: str | Path = LOCK_PATH) -> Path:
    body = build_lock(m)
    Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return Path(path)


def main() -> int:
    m = load_manifest(DEFAULT_XLSX)
    problems = validate(m)
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problems):")
        for p in problems[:50]:
            print("  -", p)
        return 1
    path = write_lock(m)
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"locked {body['n_runs']} runs -> {path}")
    print(f"manifest_lock_hash = {body['manifest_lock_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
