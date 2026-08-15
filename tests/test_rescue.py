import json

from mgr.analysis.rescue import rescue_analysis


def _write_arm(run_dir, run_id, condition, seed, ems):
    """ems: {qid: 0/1}. Writes the per-run record + items.jsonl."""
    per_run = run_dir / "results" / "arms" / "per-run"
    per_run.mkdir(parents=True, exist_ok=True)
    (per_run / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "condition": condition, "seed": seed}), encoding="utf-8"
    )
    items_dir = per_run / run_id
    items_dir.mkdir(exist_ok=True)
    (items_dir / "items.jsonl").write_text(
        "\n".join(json.dumps({"qid": q, "em": e, "gold": "A", "answer_norm": "A"}) for q, e in ems.items()),
        encoding="utf-8",
    )


def test_rescue_and_break_counts(tmp_path):
    # No-RAG: right on q1,q2; wrong on q3,q4
    _write_arm(tmp_path, "R1", "No-RAG", 42, {"q1": 1, "q2": 1, "q3": 0, "q4": 0})
    # RAG arm: rescues q3 (was wrong -> right), breaks q1 (was right -> wrong)
    _write_arm(tmp_path, "R2", "Hybrid-CARRF", 42, {"q1": 0, "q2": 1, "q3": 1, "q4": 0})

    rows = rescue_analysis(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["condition"] == "Hybrid-CARRF"
    assert r["base_wrong"] == 2 and r["rescues"] == 1 and r["rescue_rate"] == 0.5
    assert r["base_right"] == 2 and r["breaks"] == 1 and r["break_rate"] == 0.5
    assert r["net_items"] == 0 and r["net_acc_delta"] == 0.0
    assert r["arm_acc"] == 0.5 and r["base_acc"] == 0.5


def test_pools_across_seeds_and_ranks_by_net(tmp_path):
    # two seeds for the baseline and one arm; a clearly-positive and a clearly-negative arm
    _write_arm(tmp_path, "R1", "No-RAG", 42, {"q1": 0, "q2": 0})
    _write_arm(tmp_path, "R2", "No-RAG", 7, {"q1": 0, "q2": 0})
    _write_arm(tmp_path, "Rgood", "Dense-MedCPT", 42, {"q1": 1, "q2": 1})
    _write_arm(tmp_path, "Rgood2", "Dense-MedCPT", 7, {"q1": 1, "q2": 0})
    _write_arm(tmp_path, "Rbad", "Graph-only", 42, {"q1": 0, "q2": 0})
    _write_arm(tmp_path, "Rbad2", "Graph-only", 7, {"q1": 0, "q2": 0})

    rows = rescue_analysis(tmp_path)
    by = {r["condition"]: r for r in rows}
    # Dense pooled over 2 seeds x 2 items = 4 baseline-wrong, 3 rescued
    assert by["Dense-MedCPT"]["n_items"] == 4 and by["Dense-MedCPT"]["rescues"] == 3
    assert by["Graph-only"]["rescues"] == 0
    # ranked by net_items desc -> Dense first
    assert rows[0]["condition"] == "Dense-MedCPT"


def test_missing_baseline_returns_empty(tmp_path):
    _write_arm(tmp_path, "R2", "Hybrid-CARRF", 42, {"q1": 1})
    assert rescue_analysis(tmp_path, baseline="No-RAG") == []


def test_only_shared_items_compared(tmp_path):
    # arm ran a different qid set; only the overlap counts
    _write_arm(tmp_path, "R1", "No-RAG", 42, {"q1": 0, "q2": 0, "q3": 1})
    _write_arm(tmp_path, "R2", "BM25", 42, {"q1": 1, "q9": 1})  # q9 not in baseline, q2/q3 absent here
    rows = rescue_analysis(tmp_path)
    assert rows[0]["n_items"] == 1 and rows[0]["rescues"] == 1  # only q1 shared
