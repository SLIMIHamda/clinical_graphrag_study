"""Smoke harness — the Step 1 (gate H2) go/no-go.

Runs a small end-to-end slice on **No-RAG + BM25** for one benchmark/seed
(Doc 00 Step 1 exit criteria). On success the operator flips
``configs/gates.yaml`` H2 -> satisfied, which unblocks the baseline rows.

``run_smoke`` is the testable core (inject a fake client + fixtures); ``main``
is the cloud entrypoint that talks to the local vLLM server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from manifest.manifest import Manifest, load_manifest
from mgr.generate.executor import GenClient, RAGExecutor
from mgr.retrieval.factory import build_retriever
from mgr.runner import Runner
from mgr.tracking import record as rec

SMOKE_CONDITIONS = ("No-RAG", "BM25")


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_smoke(
    manifest: Manifest,
    client: GenClient,
    *,
    data_root: str | Path,
    corpus_records: list[dict[str, Any]],
    benchmark: str = "MMLU-Med",
    seed: int = 42,
    backbone: str = "Llama-70B",
    n_items: int = 200,
    results_root: str | Path = "results/smoke",
) -> list[rec.RunRecord]:
    """Execute the No-RAG and BM25 rows for one (benchmark, seed)."""
    ledger = {"H2": True, "G3": False, "P3": False}  # H2 self-gated for the smoke
    runner = Runner(manifest=manifest, gate_ledger=ledger, results_root=results_root)

    out: list[rec.RunRecord] = []
    for cond in SMOKE_CONDITIONS:
        row = next(
            (
                r
                for r in manifest.runs
                if r.condition == cond
                and r.benchmark == benchmark
                and r.seed == seed
                and r.backbone == backbone
            ),
            None,
        )
        if row is None:
            raise LookupError(f"no manifest row for {cond}/{benchmark}/{backbone}/s{seed}")
        retriever = build_retriever(cond, corpus_records if cond == "BM25" else None)
        execu = RAGExecutor(client=client, data_root=data_root, retriever=retriever, n_items=n_items)
        record = runner.run_one(row.run_id, executor=execu)
        if record is None:
            raise RuntimeError(f"{row.run_id} ({cond}) did not run (already Done or blocked?)")
        out.append(record)
    runner.rollup()
    return out


def report(records: list[rec.RunRecord]) -> bool:
    """Print a smoke report; return True iff every arm completed cleanly."""
    ok = True
    print("=== smoke report ===")
    for r in records:
        g = r.metrics.get("generation", {})
        status_ok = r.status == "Done"
        ok = ok and status_ok
        print(
            f"  {r.condition:<8} {r.benchmark:<10} n={r.n_items:<4} "
            f"acc={g.get('accuracy', float('nan')):.3f} cov={g.get('coverage', float('nan')):.3f} "
            f"tok={r.tokens['total']:<8} status={r.status}"
        )
    print("PASS" if ok else "FAIL")
    if ok:
        print("Next: set gates.H2.satisfied: true in configs/gates.yaml to unblock baseline rows.")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 1 smoke (No-RAG + BM25) -> gate H2")
    ap.add_argument("--data-root", required=True, help="dir of {benchmark}.jsonl files")
    ap.add_argument("--corpus", required=True, help="BM25 corpus JSONL ({id, text})")
    ap.add_argument("--base-url", default="http://localhost:8000", help="vLLM OpenAI-compatible URL")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--benchmark", default="MMLU-Med")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--results-root", default="results/smoke")
    args = ap.parse_args(argv)

    from mgr.clients.vllm import VLLMClient

    m = load_manifest()
    client = VLLMClient(base_url=args.base_url, api_key=args.api_key)
    records = run_smoke(
        m,
        client,
        data_root=args.data_root,
        corpus_records=load_corpus(args.corpus),
        benchmark=args.benchmark,
        seed=args.seed,
        n_items=args.n_items,
        results_root=args.results_root,
    )
    return 0 if report(records) else 1


if __name__ == "__main__":
    sys.exit(main())
