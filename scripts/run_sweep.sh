#!/usr/bin/env bash
# run_sweep.sh — run the sweep over all Ready rows (gates from configs/gates.yaml).
# Intended to run inside with_pod.sh so the GPU is torn down afterward:
#   bash infra/runpod/with_pod.sh bash scripts/run_sweep.sh
#
# Reads env (.env): VLLM_BASE_URL, NIM_API_KEY/NIM_BASE_URL, DATA_ROOT, CORPUS,
# DENSE_EMB/DENSE_IDS. Touches a heartbeat so the idle guard knows we're alive.
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

HEARTBEAT="${HEARTBEAT:-/vol/results/.heartbeat}"
mkdir -p "$(dirname "$HEARTBEAT")"

# heartbeat loop in the background so pod_watch never tears down an active sweep
( while true; do touch "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

python -m mgr.sweep --results-root "${RESULTS_ROOT:-/vol/results}"
python -m mgr.sync_status --results-root "${RESULTS_ROOT:-/vol/results}" || true
