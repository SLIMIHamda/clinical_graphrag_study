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
from mgr.gate import features as gate_features
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
class GateCapture:
    """Config for Gate-A pre-retrieval feature capture (docs/gate_a_phase_a.md).

    Off by default. Only the No-RAG arm (a ``NullRetriever``) captures — its
    generation is the base model answering with no context, which is exactly the
    signal Gate A decides on. RAG arms are untouched, so capture never perturbs
    the measured conditions. When enabled, the generation client must expose
    ``complete_text_logprobs`` (VLLMClient does).
    """

    enabled: bool = False
    n_samples: int = 5              # self-consistency draws; 0 disables the sc_* features
    sample_temperature: float = 0.7
    top_logprobs: int = 10          # candidates requested at the answer-token position
    embed_client: Any | None = None  # anything exposing .embeddings(model, [text]) -> {"data":[{"embedding": [...]}]}
    embed_model: str = ""            # empty disables the q_emb feature


@dataclass
class RAGExecutor:
    client: GenClient
    data_root: str | Path
    retriever: Retriever = None  # type: ignore[assignment]
    n_items: int | None = None   # smoke override; None = full benchmark
    # Retrieval-query cap. Embedding/reranking endpoints reject over-long input
    # (`HTTP 400: Input length 8850 exceeds maximum allowed token size 4096`),
    # which used to sink the item, score it wrong, and fail the whole run. The
    # cap applies to the *retrieval query only* — the generation prompt always
    # gets the full question, so answer quality is untouched.
    max_query_chars: int = 6000
    gate: GateCapture | None = None  # Gate-A feature capture (No-RAG arm only)

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = NullRetriever()

    def _capturing(self) -> bool:
        return bool(self.gate and self.gate.enabled and isinstance(self.retriever, NullRetriever))

    def _gate_features(
        self,
        it: Any,
        msgs: list[dict[str, str]],
        model_id: str,
        decoding: dict[str, Any],
        greedy_norm: str | None,
        lp_content: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Pre-retrieval Gate-A features for one No-RAG item: option-logprob
        confidence/entropy/margin, self-consistency over ``n_samples`` stochastic
        draws, structural stats, and an optional question embedding. Returns the
        flat feature keys (mgr.gate.features) plus ``gate_cost_tokens`` — the
        self-consistency spend, kept off the arm's token total so the gate's cost
        is measurable, not hidden inside the No-RAG accuracy record.
        """
        gate = self.gate
        assert gate is not None
        letters = list(it.options.keys()) if it.options else ["A", "B", "C", "D"]
        top = gate_features.answer_top_logprobs(lp_content or [], letters)
        option_probs = gate_features.option_probs_from_logprobs(top, letters)

        samples: list[str] = []
        g_in = g_out = 0
        base_seed = int(decoding.get("seed", 0))
        for j in range(max(0, gate.n_samples)):
            params = {**decoding, "temperature": gate.sample_temperature, "seed": base_seed * 1000 + j + 1}
            try:
                s_raw, s_usage = self.client.complete_text(model_id, msgs, **params)
            except Exception:
                continue  # a flaky sample just lowers k for this item, never sinks it
            g_in += int(s_usage.get("in", 0))
            g_out += int(s_usage.get("out", 0))
            s_norm = normalize(s_raw, it.answer_type)
            if s_norm:
                samples.append(s_norm)

        q_emb = None
        if gate.embed_client is not None and gate.embed_model:
            try:
                resp = gate.embed_client.embeddings(gate.embed_model, [it.question])
                q_emb = resp["data"][0]["embedding"]
            except Exception:
                q_emb = None

        feats = gate_features.compute_features(
            option_probs=option_probs,
            greedy_answer=greedy_norm or "",
            question=it.question,
            n_options=len(letters),
            samples=samples,
            q_emb=q_emb,
        )
        feats["gate_cost_tokens"] = {"in": g_in, "out": g_out}
        return feats

    def __call__(self, row: RunRow, cfg: dict[str, Any]) -> ExecResult:
        # Fail loud *before* a whole arm runs: capturing needs the logprob path.
        if self._capturing() and not hasattr(self.client, "complete_text_logprobs"):
            raise RuntimeError(
                "Gate-A feature capture is enabled but the generation client has no "
                "complete_text_logprobs(); add it (see docs/gate_a_phase_a.md section 3)."
            )
        benchmark_type = str(cfg["benchmark"].get("type", "MCQ"))
        model_id = str(cfg["backbone"]["model_id"])
        # The manifest hands us the contract model_id (e.g. the 70B), but a
        # pinned client (the POC's NimGenerationClient) may serve a different,
        # cheaper model regardless. Record whatever the client will actually use
        # so the run-record's gen_model is truthful, not the manifest's wish.
        gen_model = str(getattr(self.client, "model", "") or model_id)
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
        item_errors: list[str] = []
        capture = self._capturing()

        for it in items:
            rr = RetrievalResult()
            raw, usage, latency = "", {"in": 0, "out": 0}, 0.0
            lp_content: list[dict[str, Any]] | None = None
            msgs: list[dict[str, str]] = []
            item_error: str | None = None
            try:
                rr = self.retriever.retrieve(it.question[: self.max_query_chars], depth_k=depth_k)
                msgs = prompts.build_messages(
                    it.question, it.answer_type, options=it.options, context=rr.context
                )
                t0 = time.time()
                if capture:
                    raw, usage, lp_content = self.client.complete_text_logprobs(
                        model_id, msgs, top_logprobs=self.gate.top_logprobs, **decoding
                    )
                else:
                    raw, usage = self.client.complete_text(model_id, msgs, **decoding)
                latency = time.time() - t0
            except Exception as e:  # one bad item must not sink the whole run
                item_error = f"{type(e).__name__}: {e}"
                item_errors.append(f"{it.qid}: {item_error}")

            norm = normalize(raw, it.answer_type)
            tok_in += int(usage.get("in", 0))
            tok_out += int(usage.get("out", 0))
            correct = exact_match(norm, it.gold)
            row_out: dict[str, Any] = {
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
                "error": item_error,
            }
            if capture and item_error is None:
                row_out["no_rag_correct"] = int(bool(correct))
                row_out.update(self._gate_features(it, msgs, model_id, decoding, norm, lp_content))
            out_items.append(row_out)
            preds.append(norm)
            golds.append(it.gold)

        s = score(preds, golds)
        # An item that raised is scored as *wrong*, not skipped, so a transport
        # blip is otherwise indistinguishable from a model mistake. Surface the
        # count next to the accuracy it depressed.
        metrics = {
            "generation": {
                "accuracy": s.accuracy,
                "em": s.em,
                "f1": s.f1,
                "coverage": s.coverage,
                "n_item_errors": len(item_errors),
            }
        }
        # Keep every failure, not just the last one to be assigned.
        error = None
        if item_errors:
            head = "; ".join(item_errors[:3])
            more = f" (+{len(item_errors) - 3} more)" if len(item_errors) > 3 else ""
            error = f"{len(item_errors)}/{len(items)} items failed -> {head}{more}"
        return ExecResult(
            n_items=len(items),
            tokens_in=tok_in,
            tokens_out=tok_out,
            metrics=metrics,
            items=out_items,
            error=error,
            expected_n_items=expected_n_items,
            gen_model=gen_model,
        )
