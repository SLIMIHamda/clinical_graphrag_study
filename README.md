# medgraphrag-journal

Hybrid Graph-RAG for Medical QA — journal upgrade. Runner, manifest, and
statistics harness for the experimental program (RGD, CA-RRF, CARe).

**Source of truth:** `manifest/Experiment-Matrix.xlsx` › `Run_Manifest`
(244 runs, R0001–R0244). The xlsx is the *human* contract (what to run);
`manifest/manifest.lock.json` is the *machine* contract the runner executes,
frozen and hashed per sweep.

## Status

Build **Step 1 complete and smoke-ready** (Doc 00 §7) — the full No-RAG + BM25
end-to-end path that sets gate H2. **52 tests pass.** All offline-validated; no
GPU/API needed until the cloud smoke.

| Module | Role |
|---|---|
| `mgr/ids.py` | `run_id ↔ slug`, `config_hash`, result-path resolver |
| `mgr/state.py` | run state machine, gate-driven readiness, claim locks |
| `manifest/manifest.py` | load + validate the workbook; merge effective configs; cost model |
| `manifest/lock.py` | freeze xlsx → `manifest.lock.json` (resolved params + hashes) |
| `mgr/runner.py` | one row → claim → execute → record; resume + integrity |
| `mgr/tracking/{record,store}.py` | run-record schema; JSON + per-item JSONL + Parquet; DuckDB views |
| `mgr/clients/{openai_compat,nim,vllm}.py` | rate-limited clients; NIM refuses generation; vLLM for 70B |
| `mgr/generate/{prompts,extract,executor}.py` | frozen prompts; answer normalization; the generate→score loop |
| `mgr/metrics/{cost,generation,retrieval}.py` | cost meter; EM/F1/accuracy; Recall@k/Precision@k/MRR/nDCG |
| `mgr/metrics/grounding_ragas.py` | real RAGAS: faithfulness, answer relevance, context precision/recall |
| `mgr/stats/rgd.py` | **RGD** (C1): retrieval-gain vs generation-gain decomposition |
| `mgr/eval/answer_format_audit.py` | gates EM/F1 across arms (kills the EM 0→0.76 artifact) |
| `mgr/retrieval/{base,bm25,factory}.py` | retriever interface; Okapi BM25; condition→retriever wiring |
| `mgr/retrieval/{rrf,ca_rrf}.py` | RRF fusion + **CA-RRF** (C2): concept-overlap list, isolable ablation |
| `mgr/retrieval/{dense,fusion}.py` | dense (MedCPT, precomputed-ready); FusionRetriever for every hybrid arm |
| `mgr/rerank/{care_gate,cross_encoder}.py` | **CARe** (C3): gate features, logistic fit, cost-aware rule, frontier |
| `mgr/graph/{umls,store,build,neo4j_store}.py` | UMLS grounding; chunk↔concept graph (in-memory + Neo4j); build + graph_hash |
| `mgr/retrieval/graph_retriever.py` | Graph-only + graph component (concept-expansion traversal) |
| `mgr/clients/nim_adapters.py` | real NIM embedder / reranker / judge / entity extractor |
| `mgr/sweep.py` | condition → executor assembly; `run_sweep` over Ready rows |
| `mgr/sync_status.py` | reconcile run-records → xlsx status column |
| `mgr/eval/b0_crosscheck.py` | baseline accuracy vs published MedRAG (harness gate) |
| `mgr/figures.py` | F3 RGD · F4 Pareto · F5 coverage figures |
| `mgr/data/loader.py` | benchmark JSONL loader |
| `mgr/smoke.py` + `scripts/smoke.sh` | the Step 1 go/no-go (No-RAG + BM25) → gate H2 |

The cost model reproduces the workbook exactly: 244 runs, ~1081M est. tokens,
**$591.49** base / **$887.23** at 1.5×. The runner drives the real generate→
extract→score loop for both No-RAG (closed-book) and BM25 (lexical) arms,
writing qid-keyed per-item records for the stats layer.

## Quick start

```bash
pip install -e .[dev]
pytest                 # 20 tests: id roundtrip, state machine, cost reconciliation
python -m manifest.lock  # validate the workbook and freeze manifest.lock.json
```

## Decisions in force

- **D1 seeds:** 3 everywhere `{42,123,7}` (manifest as-is).
- **D2 tracking:** Parquet + DuckDB; MLflow/W&B off.
- **D3 serving:** vLLM + AWQ-int4 on 1× A100 80GB (later step).
- **D4 governance:** Clinical-Set synthetic/clean until R1 is locked.

## Running the smoke (on the pod, once data + vLLM are up)

```bash
DATA_ROOT=/vol/data/mirage CORPUS=/vol/indices/bm25/corpus.jsonl \
  BASE_URL=http://localhost:8000 BENCHMARK=MMLU-Med N_ITEMS=200 \
  bash scripts/smoke.sh
```

A `PASS` means: deterministic run, ids match the manifest, both arms processed
200 items, metrics emitted. Then set `gates.H2.satisfied: true` in
`configs/gates.yaml` to unblock the baseline rows.

## Status: all coding complete (139 tests)

The full pipeline is implemented and tested end-to-end with injectable interfaces
+ fakes — harness, every retrieval arm (No-RAG, BM25, dense, graph, CA-RRF, all
hybrids), UMLS grounding, CARe gate, all metrics (generation/retrieval/cost/
RAGAS), answer-format audit, stats (CIs, exact-p, Holm, effect sizes, RGD), the
NIM adapters, the graph build, the sweep assembly, status sync, B0 cross-check,
and the figures. All three contributions (C1/C2/C3) are coded.

**What's left is not code — it's running it:** get a NIM key, stage data on the
volume, and follow [docs/RUNBOOK.md](docs/RUNBOOK.md). Dropping a real
`NIM_API_KEY` into `.env` activates the embedder/reranker/judge with no code
change; the A100 only runs generation.

## To run (summary — full steps in the runbook)
1. `cp .env.example .env`, add `NIM_API_KEY`, `RUNPOD_API_KEY`, `VOLUME_ID`.
2. Get QA data: `python -m mgr.data.convert_mirage --mirage benchmark.json --out-dir data`.
3. `python -m manifest.lock` (validates workbook, freezes the contract).
4. Stage corpora/embeddings to `/vol`: `bash scripts/fetch_data.sh`.
5. Smoke → H2: `bash infra/runpod/with_pod.sh bash scripts/smoke.sh`.
6. Graph → G3: `bash scripts/build_graph.sh`.
7. Sweep: `bash infra/runpod/with_pod.sh bash scripts/run_sweep.sh` (auto teardown).
