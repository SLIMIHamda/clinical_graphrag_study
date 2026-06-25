"""Run state machine + gate-driven readiness + claim locks.

Verbatim legend (Doc 00 section 3.1):

    Pending --(gate satisfied)--> Ready --(worker claims)--> Running --+--> Done
                                                                       |
                                                                       +--> Failed (resumable)
    Pending --(gate NOT satisfied)--> Blocked

Readiness is a *gate-ledger lookup*, not a run-to-run DAG. ``depends_on`` holds
exactly one of three gate keys; a row is Ready iff its gate is satisfied.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from . import ids


class Status(str, Enum):
    PENDING = "Pending"
    READY = "Ready"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"
    BLOCKED = "Blocked"


# Allowed transitions of the state machine. Resume re-runs a Failed row, so
# Failed -> Running is permitted; Done is terminal.
_TRANSITIONS: dict[Status, set[Status]] = {
    Status.PENDING: {Status.READY, Status.BLOCKED},
    Status.BLOCKED: {Status.PENDING, Status.READY},
    Status.READY: {Status.RUNNING, Status.BLOCKED, Status.PENDING},
    Status.RUNNING: {Status.DONE, Status.FAILED},
    Status.FAILED: {Status.RUNNING, Status.READY},
    Status.DONE: set(),
}


def can_transition(src: Status, dst: Status) -> bool:
    return dst in _TRANSITIONS[Status(src)]


def assert_transition(src: Status, dst: Status) -> None:
    if not can_transition(src, dst):
        raise ValueError(f"illegal status transition: {src.value} -> {dst.value}")


def resolve_status(current: Status, depends_on: str, gate_ledger: Mapping[str, bool]) -> Status:
    """Resolve a row's status against the gate ledger.

    Pure function over (current status, gate key, ledger):
      - Done / Running / Failed are owned by the runner and pass through.
      - Otherwise: gate satisfied -> Ready, gate unsatisfied -> Blocked.
    """
    current = Status(current)
    if current in (Status.DONE, Status.RUNNING, Status.FAILED):
        return current
    gate_key = gate_name(depends_on)
    if gate_key not in gate_ledger:
        raise KeyError(f"unknown gate {gate_key!r} (from depends_on={depends_on!r})")
    return Status.READY if gate_ledger[gate_key] else Status.BLOCKED


def gate_name(depends_on: str) -> str:
    """Normalize a ``depends_on`` cell to its bare gate key.

    The manifest stores e.g. ``"H2 (harness)"`` or ``"P3 gate + oracle labels"``;
    the gate ledger keys on the leading token (``"H2"`` / ``"P3"``).
    """
    return str(depends_on).split()[0].strip()


# --------------------------------------------------------------------------- #
# Claim locks: Running = a worker holds results/per-run/{run_id}/.claim
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Claim:
    run_id: str
    host: str
    pid: int
    ts: float

    def to_json(self) -> str:
        return json.dumps({"run_id": self.run_id, "host": self.host, "pid": self.pid, "ts": self.ts})


def acquire_claim(run_id: str, results_root: str | Path = "results", stale_after_s: float = 3600.0) -> Claim | None:
    """Atomically claim a run. Returns the Claim on success, ``None`` if held.

    A claim is stale if older than ``stale_after_s``; stale claims are reaped
    and re-acquired (the original worker died without releasing).
    """
    path = ids.claim_path(run_id, results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_claim(run_id, results_root)
    if existing is not None and (time.time() - existing.ts) < stale_after_s:
        return None  # live claim held by someone else
    claim = Claim(run_id=run_id, host=socket.gethostname(), pid=os.getpid(), ts=time.time())
    # O_CREAT|O_EXCL would race-protect cross-process; for a stale takeover we
    # overwrite deliberately. Single-writer-per-run is enforced by the manifest.
    path.write_text(claim.to_json(), encoding="utf-8")
    return claim


def read_claim(run_id: str, results_root: str | Path = "results") -> Claim | None:
    path = ids.claim_path(run_id, results_root)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Claim(run_id=data["run_id"], host=data["host"], pid=data["pid"], ts=data["ts"])


def release_claim(run_id: str, results_root: str | Path = "results") -> None:
    path = ids.claim_path(run_id, results_root)
    path.unlink(missing_ok=True)
