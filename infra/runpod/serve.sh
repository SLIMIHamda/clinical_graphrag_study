#!/usr/bin/env bash
# serve.sh — start the services that read /vol, on the pod:
#   - vLLM OpenAI-compatible server, AWQ-int4 70B (fits one A100, no TP) [D3]
#   - Neo4j (graph), via docker-compose
# BM25/FAISS are in-process in the runner, reading /vol/indices.
#
# Generation goes to THIS vLLM endpoint, never NIM (the nim.py guard enforces it).
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

MODEL="${VLLM_MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "[serve] starting Neo4j…"
docker compose -f infra/neo4j/docker-compose.yml up -d

echo "[serve] starting vLLM ($MODEL, AWQ-int4)…"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port "$VLLM_PORT" \
  > /vol/results/vllm.log 2>&1 &

echo "[serve] waiting for vLLM health…"
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[serve] vLLM ready on :${VLLM_PORT}"; exit 0
  fi
  sleep 10
done
echo "[serve] vLLM did not become healthy in time" >&2
exit 1
