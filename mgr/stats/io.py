"""Load per-item JSONL and align two runs by ``qid`` for paired tests.

Paired permutation tests require item-level alignment (Doc 00 section 3.3):
every condition writes the same qid set per benchmark, so we join on qid and
compare the same items across systems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_items_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def align_by_qid(
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    metric: str = "em",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return paired metric arrays over the qids present in *both* runs.

    Raises if the qid sets differ in a way that would misalign the pairing,
    surfacing a real defect rather than silently comparing different items.
    """
    a = {str(it["qid"]): it for it in items_a}
    b = {str(it["qid"]): it for it in items_b}
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("no shared qids between the two runs")
    miss_a, miss_b = set(b) - set(a), set(a) - set(b)
    if miss_a or miss_b:
        raise ValueError(
            f"qid sets differ: {len(miss_b)} only in A, {len(miss_a)} only in B; "
            "paired tests require identical item sets"
        )
    va = np.array([float(a[q][metric]) for q in shared], dtype=float)
    vb = np.array([float(b[q][metric]) for q in shared], dtype=float)
    return va, vb, shared
