"""Convert MIRAGE's benchmark.json into our per-benchmark JSONL.

MIRAGE ships one benchmark.json keyed by dataset (mmlu, medqa, medmcqa,
pubmedqa, bioasq), each a map of qid -> {question, options, answer}. We
materialize one ``{benchmark}.jsonl`` per manifest benchmark in the normalized
shape the loader reads:

    {"qid": "...", "question": "...", "options": {"A": "..."}, "answer": "B"}

The QA files are small (~tens of MB total) so this runs locally; the heavy
corpora/embeddings are fetched separately to the cloud volume (see fetch_data.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .loader import write_items_fixture

# manifest benchmark name -> MIRAGE dataset key
BENCHMARK_TO_MIRAGE = {
    "MMLU-Med": "mmlu",
    "MedQA-US": "medqa",
    "MedMCQA": "medmcqa",
    "PubMedQA": "pubmedqa",
    "BioASQ-YN": "bioasq",
}


def convert_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Map a MIRAGE dataset block to our normalized rows."""
    rows: list[dict[str, Any]] = []
    for qid, item in records.items():
        row: dict[str, Any] = {
            "qid": str(qid),
            "question": str(item.get("question", "")).strip(),
            "answer": item.get("answer"),
        }
        opts = item.get("options")
        if isinstance(opts, dict) and opts:
            row["options"] = {str(k): str(v) for k, v in opts.items()}
        rows.append(row)
    return rows


def convert_file(
    mirage_path: str | Path,
    out_dir: str | Path,
    *,
    mapping: dict[str, str] = BENCHMARK_TO_MIRAGE,
) -> dict[str, int]:
    """Write one JSONL per mapped benchmark; returns {benchmark: n_items}."""
    data = json.loads(Path(mirage_path).read_text(encoding="utf-8"))
    written: dict[str, int] = {}
    for benchmark, key in mapping.items():
        if key not in data:
            continue
        rows = convert_records(data[key])
        write_items_fixture(benchmark, out_dir, rows)
        written[benchmark] = len(rows)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MIRAGE benchmark.json -> per-benchmark JSONL")
    ap.add_argument("--mirage", required=True, help="path to MIRAGE benchmark.json")
    ap.add_argument("--out-dir", default="data", help="output dir for {benchmark}.jsonl")
    args = ap.parse_args(argv)

    written = convert_file(args.mirage, args.out_dir)
    if not written:
        print("no known MIRAGE datasets found in the input", file=sys.stderr)
        return 1
    for bench, n in sorted(written.items()):
        print(f"  {bench:<12} {n:>6} items -> {args.out_dir}/{bench}.jsonl")
    print(f"converted {len(written)} benchmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
