#!/usr/bin/env bash
# down.sh — persist deltas to /vol, then terminate the pod. IDEMPOTENT: safe to
# run twice, safe to run when no pod exists (layer 2 of 3). Results already live
# on the Network Volume, so termination loses nothing.
set -uo pipefail
[ -f .env ] && set -a && . ./.env && set +a

if [ ! -f .pod_id ]; then
  echo "[down] no .pod_id — nothing to terminate"; exit 0
fi
POD_ID="$(cat .pod_id)"

# best-effort: flush any local run-records onto the volume (no-op if already there)
sync || true

echo "[down] terminating pod $POD_ID…"
if runpodctl remove pod "$POD_ID"; then
  rm -f .pod_id
  echo "[down] terminated and cleared .pod_id"
else
  echo "[down] remove reported an error (pod may already be gone)"; rm -f .pod_id
fi
