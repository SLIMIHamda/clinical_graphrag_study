"""Idle pod guard — the external watchdog layer (Doc 00 section 6).

Three layers protect against a forgotten paid GPU: the session trap in
with_pod.sh, the idempotent down.sh, and this external watcher. The runner
touches a heartbeat file; if the pod is up but the heartbeat has gone stale past
a threshold (no active run), the guard tears the pod down.

The idle decision is pure and unit-tested; the polling ``main`` wraps it around
the RunPod API + down.sh and is intended to run on a cheap always-on host (or as
a cron), never on the GPU pod itself.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def is_idle(last_heartbeat_ts: float, now: float, idle_threshold_s: float) -> bool:
    """True iff the heartbeat is older than the idle threshold."""
    return (now - last_heartbeat_ts) > idle_threshold_s


def read_heartbeat(path: str | Path) -> float | None:
    """Last heartbeat time (file mtime), or None if never written."""
    p = Path(path)
    return p.stat().st_mtime if p.exists() else None


def teardown(down_script: str | Path) -> int:
    """Invoke the idempotent down.sh; returns its exit code."""
    return subprocess.run(["bash", str(down_script)], check=False).returncode


def check_once(heartbeat: str | Path, down_script: str | Path, *, idle_threshold_s: float, now: float | None = None) -> str:
    """One guard tick. Returns an action string for logging/testing."""
    now = time.time() if now is None else now
    hb = read_heartbeat(heartbeat)
    if hb is None:
        return "no-heartbeat-yet"
    if is_idle(hb, now, idle_threshold_s):
        teardown(down_script)
        return "torn-down"
    return "alive"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Idle GPU-pod guard")
    ap.add_argument("--heartbeat", default="/vol/results/.heartbeat")
    ap.add_argument("--down-script", default="infra/runpod/down.sh")
    ap.add_argument("--idle-threshold-s", type=float, default=1800.0)  # 30 min
    ap.add_argument("--interval-s", type=float, default=120.0)
    ap.add_argument("--once", action="store_true", help="check a single time and exit")
    args = ap.parse_args(argv)

    while True:
        action = check_once(args.heartbeat, args.down_script, idle_threshold_s=args.idle_threshold_s)
        print(f"[pod_watch] {action}", flush=True)
        if action == "torn-down" or args.once:
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    sys.exit(main())
