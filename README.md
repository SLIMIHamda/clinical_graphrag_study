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
| `mgr/stats/rgd.py` | **RGD** (C1): retrieval-gain vs generation-gain decomposition |
| `mgr/eval/answer_format_audit.py` | gates EM/F1 across arms (kills the EM 0→0.76 artifact) |
| `mgr/retrieval/{base,bm25,factory}.py` | retriever interface; Okapi BM25; condition→retriever wiring |
| `mgr/retrieval/{rrf,ca_rrf}.py` | RRF fusion + **CA-RRF** (C2): concept-overlap list, isolable ablation |
| `mgr/rerank/{care_gate,cross_encoder}.py` | **CARe** (C3): gate features, logistic fit, cost-aware rule, frontier |
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

## Next build steps

- **B0 cross-check:** validate No-RAG/BM25 metrics against published MedRAG.
- **Step 2 (gate G3):** `mgr/graph/` — Neo4j build, chunk-level anchors, UMLS
  grounding + coverage curve; freeze `graph_hash`. Unblocks graph + hybrid arms.
- **Step 3–5:** dense (MedCPT), RRF/CA-RRF fusion, CARe gate, model zoo.
- **Stats:** `mgr/stats/` — bootstrap CIs, ≥100k-perm exact p, Holm, effect size.
- **Infra:** `infra/runpod/` lifecycle scripts + idle pod guard.
