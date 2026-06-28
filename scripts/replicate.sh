#!/usr/bin/env bash
# replicate.sh — one-command replication (H4). Reproduces the study from the
# frozen artifacts: pinned env, manifest lock, seeds, configs, index hashes.
#
# Resume-safe: skips Done rows and reuses content-addressed caches, so a partial
# prior run is continued, not redone.
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

echo "[replicate] 1/4 install pinned environment"
pip install -e .[dev] -q

echo "[replicate] 2/4 validate workbook + freeze manifest.lock.json"
python -m manifest.lock

echo "[replicate] 3/4 stage data to the volume (idempotent)"
bash scripts/fetch_data.sh

echo "[replicate] 4/4 run the sweep on the A100 (auto teardown), then sync status"
bash infra/runpod/with_pod.sh bash scripts/run_sweep.sh

echo "[replicate] done — results in ${RESULTS_ROOT:-/vol/results}"
