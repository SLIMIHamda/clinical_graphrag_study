"""RAGExecutor — the real per-item generate -> extract -> score loop.

Plugs into the runner as an ``Executor`` (one row -> ExecResult). Identical for
every condition; the only thing that varies is the injected ``retriever`` (Null
for No-RAG, BM25/dense/graph/fusion otherwise). This keeps prompt parity and
makes the condition the *only* manipulated factor.

Per-item rows follow the Doc 00 section 3.3 schema (qid-keyed for the paired
permutation tests). The cross-arm answer-format audit runs later in the stats
layer over these per-item labels — never inside a single run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mgr.data.loader import load_items
from mgr.generate import prompts
from mgr.generate.extract import normalize
from mgr.metrics.generation import exact_match, score, token_f1
from mgr.retrieval.base import NullRetriever, RetrievalResult, Retriever
from mgr.runner import ExecResult
from manifest.manifest import RunRow


class GenClient:
    """Minimal generation-client protocol (satisfied by VLLMClient)."""

    def complete_text(self, model: str, messages: list[dict[str, str]], **params: Any) -> tuple[str, dict[str, int]]:
        ...


@dataclass
class RAGExecutor:
    client: GenClient
    data_root: str | Path
    retriever: Retriever = None  # type: ignore[assignment]
    n_items: int | None = None   # smoke override; None = full benchmark

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = NullRetriever()

    def __call__(self, row: RunRow, cfg: dict[str, Any]) -> ExecResult:
        benchmark_type = str(cfg["benchmark"].get("type", "MCQ"))
        model_id = str(cfg["backbone"]["model_id"])
        depth_k = int(row.retr_depth_k) if str(row.retr_depth_k).isdigit() else 10
        decoding = dict(cfg["base"].get("decoding", {}))
        decoding.pop("seed_is_authoritative", None)
        decoding["seed"] = row.seed  # deterministic decoding per the run's seed

        items = load_items(row.benchmark, self.data_root, benchmark_type=benchmark_type, n_items=self.n_items)
        # Intended count: the smoke subset size, else the benchmark's declared
        # n_items (so a truncated data file is caught by the integrity check).
        declared = int(cfg["benchmark"].get("n_items", len(items)))
        expected_n_items = self.n_items if self.n_items is not None else declared

        out_items: list[dict[str, Any]] = []
        preds: list[str | None] = []
        golds: list[str | None] = []
        tok_in = tok_out = 0
        error: str | None = None

        for it in items:
            rr = RetrievalResult()
            raw, usage, latency = "", {"in": 0, "out": 0}, 0.0
            try:
                rr = self.retriever.retrieve(it.question, depth_k=depth_k)
                msgs = prompts.build_messages(
                    it.question, it.answer_type, options=it.options, context=rr.context
                )
                t0 = time.time()
                raw, usage = self.client.complete_text(model_id, msgs, **decoding)
                latency = time.time() - t0
            except Exception as e:  # one bad item must not sink the whole run
                error = f"{type(e).__name__}: {e}"

            norm = normalize(raw, it.answer_type)
            tok_in += int(usage.get("in", 0))
            tok_out += int(usage.get("out", 0))
            correct = exact_match(norm, it.gold)
            out_items.append(
                {
                    "qid": it.qid,
                    "retrieved_ids": rr.retrieved_ids,
                    "ranks": rr.ranks,
                    "rerank_fired": rr.rerank_fired,
                    "answer_raw": raw,
                    "answer_norm": norm,
                    "gold": it.gold,
                    "correct": bool(correct),
                    "em": correct,
                    "f1": token_f1(norm, it.gold),
                    "tokens": {"in": int(usage.get("in", 0)), "out": int(usage.get("out", 0))},
                    "latency_s": latency,
                }
            )
            preds.append(norm)
            golds.append(it.gold)

        s = score(preds, golds)
        metrics = {
            "generation": {
                "accuracy": s.accuracy,
                "em": s.em,
                "f1": s.f1,
                "coverage": s.coverage,
            }
        }
        return ExecResult(
            n_items=len(items),
            tokens_in=tok_in,
            tokens_out=tok_out,
            metrics=metrics,
            items=out_items,
            error=error,
            expected_n_items=expected_n_items,
        )
