#!/usr/bin/env bash
# Step 1 smoke (No-RAG + BM25) -> gate H2. Run from the repo root on the pod
# (or any host with the vLLM endpoint reachable and the data materialized).
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/vol/data/mirage}"
CORPUS="${CORPUS:-/vol/indices/bm25/corpus.jsonl}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
BENCHMARK="${BENCHMARK:-MMLU-Med}"
N_ITEMS="${N_ITEMS:-200}"

python -m mgr.smoke \
  --data-root "$DATA_ROOT" \
  --corpus "$CORPUS" \
  --base-url "$BASE_URL" \
  --benchmark "$BENCHMARK" \
  --n-items "$N_ITEMS" \
  --results-root "results/smoke"
