#!/usr/bin/env bash
# run_local.sh — run the sweep on a self-hosted GPU box.
#
# The counterpart to run_sweep.sh, which assumes a RunPod pod and wraps itself in
# with_pod.sh for teardown. Here the machine is yours and stays up, so there is
# no pod lifecycle, no idle guard and no heartbeat — but there is a pre-flight,
# because the failure this replaces is a sweep that wires two arms out of six and
# still reports every row it ran as Done.
#
# Reads .env. For a run over credentialed chart data, leave NIM_API_KEY unset:
# every component then resolves to its local implementation and no text leaves
# the machine.
#
#   bash scripts/run_local.sh                 # pre-flight, then run
#   PREFLIGHT_ONLY=1 bash scripts/run_local.sh
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

RESULTS_ROOT="${RESULTS_ROOT:-results/local}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000}"
GEN_BACKEND="${GEN_BACKEND:-vllm}"
STRICT="${STRICT:-1}"

if [ "${GEN_BACKEND}" = "vllm" ]; then
  echo "[run_local] waiting for vLLM at ${VLLM_BASE_URL} ..."
  for _ in $(seq 1 "${VLLM_WAIT_TRIES:-60}"); do
    if curl -sf "${VLLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
      echo "[run_local] vLLM is up"
      break
    fi
    sleep 5
  done
  curl -sf "${VLLM_BASE_URL}/v1/models" >/dev/null 2>&1 || {
    echo "[run_local] ERROR: vLLM never answered at ${VLLM_BASE_URL}" >&2
    echo "  start it with: vllm serve \"\${HF_MODEL_ID}\" --max-model-len 8192 --port 8000" >&2
    exit 1
  }
fi

if [ -n "${NIM_API_KEY:-}" ]; then
  echo "[run_local] NOTE: NIM_API_KEY is set — embeddings/rerank/judge may use the hosted API."
  echo "           Unset it for runs over credentialed data."
fi

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  exec python -m mgr.sweep --preflight-only --results-root "${RESULTS_ROOT}"
fi

STRICT_FLAG=""
[ "${STRICT}" = "1" ] && STRICT_FLAG="--strict"

mkdir -p "${RESULTS_ROOT}"
python -m mgr.sweep ${STRICT_FLAG} --results-root "${RESULTS_ROOT}"
python -m mgr.sync_status --results-root "${RESULTS_ROOT}" || true
python -m mgr.analysis.rescue --run-dir "${RESULTS_ROOT}" || true

echo "[run_local] done -> ${RESULTS_ROOT}"
echo "  Phase-A: python -m mgr.analysis.gate_signal \\"
echo "      ${RESULTS_ROOT}/arms/per-run/<No-RAG run>/items.jsonl \\"
echo "      --run-dir ${RESULTS_ROOT} --arm BM25 --lambda 0.02"
