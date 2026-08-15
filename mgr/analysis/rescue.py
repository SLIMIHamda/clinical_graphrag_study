"""Retrieval rescue/break analysis: does RAG help where the model is *wrong*?

The powered MedQA run showed every RAG arm below No-RAG on average. That average
hides the question that matters: on the items No-RAG gets **wrong** (i.e. the model
does not already know the answer), does retrieval rescue any? And on the items
No-RAG gets **right**, how many does retrieval **break** (distraction)?

For each arm vs the No-RAG baseline, over items both ran (keyed by seed+qid):
  rescue_rate = P(arm correct | No-RAG wrong)   -- retrieval's upside
  break_rate  = P(arm wrong   | No-RAG right)   -- retrieval's downside (distraction)
  net_items   = rescues - breaks                -- net items moved
A method that helps on hard items but a distractor on easy ones shows high rescue
AND high break -- which is exactly the retrieval-generation divergence C1 names.

Reads a run directory's per-run records (same layout as mgr.report). Pure stdlib.

CLI:  python -m mgr.analysis.rescue <run_dir> [--baseline No-RAG]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PER_RUN_GLOB = "results/arms/per-run/R*.json"


@dataclass
class ArmRescue:
    condition: str
    n_items: int          # items compared against the baseline (pooled over seeds)
    arm_acc: float
    base_acc: float
    base_wrong: int       # how many baseline got wrong (retrieval's opportunity)
    rescues: int          # of base_wrong, how many the arm got right
    rescue_rate: float
    base_right: int
    breaks: int           # of base_right, how many the arm broke
    break_rate: float
    net_items: int        # rescues - breaks
    net_acc_delta: float  # net_items / n_items


def _load_arm_items(run_dir: Path) -> dict[str, dict[tuple[int, str], int]]:
    """condition -> {(seed, qid): em(0/1)} pooled across seeds."""
    run = Path(run_dir)
    by_cond: dict[str, dict[tuple[int, str], int]] = {}
    for rec_path in sorted(run.glob(PER_RUN_GLOB)):
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cond = rec.get("condition")
        seed = rec.get("seed")
        items_file = rec_path.with_suffix("") / "items.jsonl"  # R0001.json -> R0001/items.jsonl
        if cond is None or not items_file.exists():
            continue
        bucket = by_cond.setdefault(cond, {})
        for line in items_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                it = json.loads(line)
            except ValueError:
                continue
            qid = it.get("qid")
            if qid is None:
                continue
            bucket[(seed, str(qid))] = int(bool(it.get("em")))
    return by_cond


def rescue_analysis(run_dir: str | Path, *, baseline: str = "No-RAG") -> list[dict[str, Any]]:
    """Per-arm rescue/break stats vs ``baseline``, most net-positive first.

    Returns [] if the baseline arm's per-item data is absent (e.g. a run that
    stopped before writing items) so callers can degrade gracefully.
    """
    by_cond = _load_arm_items(Path(run_dir))
    base = by_cond.get(baseline)
    if not base:
        return []

    out: list[ArmRescue] = []
    for cond, items in by_cond.items():
        if cond == baseline:
            continue
        keys = [k for k in items if k in base]  # only items both arms ran
        if not keys:
            continue
        wrong = [k for k in keys if base[k] == 0]
        right = [k for k in keys if base[k] == 1]
        rescues = sum(1 for k in wrong if items[k] == 1)
        breaks = sum(1 for k in right if items[k] == 0)
        out.append(
            ArmRescue(
                condition=cond,
                n_items=len(keys),
                arm_acc=round(sum(items[k] for k in keys) / len(keys), 4),
                base_acc=round(sum(base[k] for k in keys) / len(keys), 4),
                base_wrong=len(wrong),
                rescues=rescues,
                rescue_rate=round(rescues / len(wrong), 4) if wrong else 0.0,
                base_right=len(right),
                breaks=breaks,
                break_rate=round(breaks / len(right), 4) if right else 0.0,
                net_items=rescues - breaks,
                net_acc_delta=round((rescues - breaks) / len(keys), 4),
            )
        )
    out.sort(key=lambda a: a.net_items, reverse=True)
    return [asdict(a) for a in out]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m mgr.analysis.rescue <run_dir> [--baseline No-RAG]")
        return 0
    run_dir = argv[0]
    baseline = argv[argv.index("--baseline") + 1] if "--baseline" in argv else "No-RAG"
    rows = rescue_analysis(run_dir, baseline=baseline)
    if not rows:
        print(f"no rescue analysis: baseline {baseline!r} per-item data not found under {run_dir}")
        return 1
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
