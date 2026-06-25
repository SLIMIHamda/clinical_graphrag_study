"""Identity contract: run_id <-> slug, config hashing, result-path resolution.

Design invariants (Doc 00 section 0):
- ``run_id`` (R0001..R0244) is the canonical, sequential key. It is never
  composite and never re-derived from the dimensions.
- The human slug ``{condition}__{benchmark}__{backbone}__s{seed}`` is *derived*
  from a row, never primary. It exists for logs and filenames only.
- ``config_hash`` is the sha256 of the merged, canonicalized effective config.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

RUN_ID_RE = re.compile(r"^R(\d{4})$")


def run_id_index(run_id: str) -> int:
    """Return the integer index of a canonical run id (``R0123`` -> 123).

    Raises ``ValueError`` if the id is not the canonical ``R####`` form.
    """
    m = RUN_ID_RE.match(run_id)
    if not m:
        raise ValueError(f"not a canonical run_id: {run_id!r} (expected R####)")
    return int(m.group(1))


def format_run_id(index: int) -> str:
    """Inverse of :func:`run_id_index` (123 -> ``R0123``)."""
    if index < 0:
        raise ValueError(f"run index must be non-negative, got {index}")
    return f"R{index:04d}"


def slug(condition: str, benchmark: str, backbone: str, seed: int | str) -> str:
    """Derive the canonical human slug for a run.

    The slug is purely cosmetic. It must round-trip its four components but is
    never used as a primary key (that is always ``run_id``).
    """
    return f"{condition}__{benchmark}__{backbone}__s{seed}"


def parse_slug(value: str) -> dict[str, str]:
    """Split a slug back into its four components (round-trip with :func:`slug`)."""
    parts = value.split("__")
    if len(parts) != 4 or not parts[3].startswith("s"):
        raise ValueError(f"malformed slug: {value!r}")
    return {
        "condition": parts[0],
        "benchmark": parts[1],
        "backbone": parts[2],
        "seed": parts[3][1:],
    }


def _canonical(obj: Any) -> Any:
    """Recursively canonicalize a config for hashing.

    Mappings are key-sorted; sequences are preserved in order. This makes the
    hash insensitive to dict key ordering but sensitive to list ordering (which
    is semantically meaningful, e.g. retriever order).
    """
    if isinstance(obj, Mapping):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def canonical_json(config: Mapping[str, Any]) -> str:
    """Serialize a config to a deterministic canonical JSON string."""
    return json.dumps(_canonical(config), separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def config_hash(config: Mapping[str, Any]) -> str:
    """sha256 of the merged, canonicalized effective config."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def result_dir(run_id: str, results_root: str | Path = "results") -> Path:
    """Per-run result directory: ``results/per-run/{run_id}/``."""
    return Path(results_root) / "per-run" / run_id


def record_path(run_id: str, results_root: str | Path = "results") -> Path:
    """Run-record path: ``results/per-run/{run_id}.json``."""
    return Path(results_root) / "per-run" / f"{run_id}.json"


def items_path(run_id: str, results_root: str | Path = "results") -> Path:
    """Per-item JSONL path: ``results/per-run/{run_id}/items.jsonl``."""
    return result_dir(run_id, results_root) / "items.jsonl"


def claim_path(run_id: str, results_root: str | Path = "results") -> Path:
    """Claim-lock path: ``results/per-run/{run_id}/.claim``."""
    return result_dir(run_id, results_root) / ".claim"
