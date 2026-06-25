import pytest

from manifest import lock
from manifest.manifest import budget_summary, load_manifest, validate
from mgr.state import Status, resolve_status


@pytest.fixture(scope="module")
def m():
    return load_manifest()


def test_manifest_loads_244_runs(m):
    assert len(m) == 244
    assert m.runs[0].run_id == "R0001"


def test_validation_clean(m):
    problems = validate(m)
    assert problems == [], "\n".join(problems)


def test_cost_model_reconciles_with_sheet(m):
    # The strongest cross-check: recompute est_tokens/est_cost for every row and
    # confirm they match the workbook's own columns (Doc 00 cost model).
    for r in m.runs:
        assert abs(r.computed_est_tokens() - r.est_tokens) <= 1.0
        assert abs(r.computed_est_cost() - r.est_cost) <= 1e-6


def test_budget_envelope_in_expected_range(m):
    b = budget_summary(m)
    # Doc 00: ~$591 base, ~$887 at 1.5x.
    assert 550 <= b["est_cost_usd"] <= 650
    assert 820 <= b["est_cost_usd_1.5x"] <= 980


def test_gate_distribution(m):
    gates = {}
    for r in m.runs:
        gates[r.gate] = gates.get(r.gate, 0) + 1
    assert set(gates) == {"H2", "G3", "P3"}


def test_readiness_follows_gate_ledger(m):
    # All Pending: with only H2 satisfied, exactly the H2 rows are Ready.
    ledger = {"H2": True, "G3": False, "P3": False}
    ready = m.resolve_ready(ledger)
    assert all(r.gate == "H2" for r in ready)
    assert len(ready) == sum(1 for r in m.runs if r.gate == "H2")


def test_effective_config_merges_all_layers(m):
    r = m.by_id("R0001")
    cfg = m.effective_config(r)
    assert set(cfg) >= {"base", "condition", "benchmark", "backbone", "seed", "run_id"}
    assert cfg["seed"] == r.seed
    assert cfg["benchmark"]["benchmark"] == r.benchmark


def test_lock_builds_and_is_stable(m):
    a = lock.build_lock(m)
    b = lock.build_lock(m)
    assert a["n_runs"] == 244
    assert a["manifest_lock_hash"] == b["manifest_lock_hash"]  # deterministic
    # every run carries a distinct config_hash per (condition,benchmark,backbone,seed)
    hashes = {e["config_hash"] for e in a["runs"]}
    assert len(hashes) == 244
