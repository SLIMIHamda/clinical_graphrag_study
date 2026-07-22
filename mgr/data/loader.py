"""Benchmark loaders (Doc 00 section 1: src/mgr/data).

A benchmark file is one JSONL per benchmark under ``data_root``:

    {"qid": "...", "question": "...", "options": {"A": "...", ...}, "answer": "B"}

This is the normalized form we materialize the MIRAGE datasets into (a converter
from MIRAGE's benchmark.json lands with the data step). The loader stays schema-
small so the smoke set and the full sweep share one path. Item order is stable
(seed-independent here); decoding seed is applied at generation, not load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mgr.generate.prompts import answer_type_for


@dataclass(frozen=True)
class BenchmarkItem:
    qid: str
    question: str
    gold: str | None
    answer_type: str
    options: dict[str, str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def benchmark_path(benchmark: str, data_root: str | Path) -> Path:
    return Path(data_root) / f"{benchmark}.jsonl"


def load_items(
    benchmark: str,
    data_root: str | Path,
    *,
    benchmark_type: str = "MCQ",
    n_items: int | None = None,
) -> list[BenchmarkItem]:
    """Load a benchmark's items. ``benchmark_type`` is the Benchmarks-sheet
    ``type`` string (mapped to an internal answer type)."""
    atype = answer_type_for(benchmark_type)
    path = benchmark_path(benchmark, data_root)
    if not path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")
    items: list[BenchmarkItem] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                BenchmarkItem(
                    qid=str(d["qid"]),
                    question=str(d["question"]),
                    gold=(None if d.get("answer") is None else str(d["answer"])),
                    answer_type=atype,
                    options=d.get("options"),
                    meta={k: v for k, v in d.items() if k not in {"qid", "question", "answer", "options"}},
                )
            )
            if n_items is not None and len(items) >= n_items:
                break
    return items


def write_items_fixture(benchmark: str, data_root: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Test/dev helper: materialize a benchmark JSONL from raw dicts."""
    path = benchmark_path(benchmark, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
