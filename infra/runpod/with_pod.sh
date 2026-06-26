#!/usr/bin/env bash
# with_pod.sh <command...> — provision an A100, attach /vol, serve, run the
# command, then ALWAYS tear down. The trap fires on EXIT/ERR/INT so a crash or
# Ctrl-C never leaves a paid GPU running (Doc 00 section 6, layer 1 of 3).
#
#   ./infra/runpod/with_pod.sh bash scripts/run_sweep.sh
#
# Requires: runpodctl configured, env from .env (RUNPOD_API_KEY, VOLUME_ID, ...).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f .env ] && set -a && . ./.env && set +a

cleanup() {
  echo "[with_pod] tearing down (trap)…"
  bash "$HERE/down.sh" || true
}
trap cleanup EXIT ERR INT TERM

bash "$HERE/up.sh"
bash "$HERE/serve.sh"

echo "[with_pod] running: $*"
"$@"
# normal exit also runs cleanup via the EXIT trap
