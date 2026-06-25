import pytest

from manifest.manifest import load_manifest
from mgr.runner import ExecResult, Runner, stub_executor
from mgr.tracking import record as rec


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture
def runner(manifest, tmp_path):
    # Only H2 satisfied -> only baseline/main rows are Ready.
    return Runner(
        manifest=manifest,
        gate_ledger={"H2": True, "G3": False, "P3": False},
        results_root=tmp_path / "results",
        manifest_lock_hash="deadbeef",
    )


def test_run_one_ready_row_produces_done_record(runner, manifest):
    # R0001 is a No-RAG row gated on H2.
    record = runner.run_one("R0001")
    assert record is not None
    assert record.status == "Done"
    assert record.executor == "stub"
    assert record.n_items == manifest.by_id("R0001").n_items
    assert record.cost_est_usd > 0
    assert record.cost_actual_usd == 0.0  # stub spends nothing
    assert rec.integrity_ok(record, manifest.by_id("R0001").n_items)


def test_blocked_row_is_skipped(runner, manifest):
    # Find a G3-gated row; with G3 unsatisfied it must not run.
    g3 = next(r for r in manifest.runs if r.gate == "G3")
    assert runner.run_one(g3.run_id) is None
    assert rec.read_record(g3.run_id, runner.results_root) is None


def test_resume_skips_done_rows(runner):
    first = runner.run_one("R0001")
    assert first is not None
    # second call sees a Done record and skips (no rerun)
    assert runner.run_one("R0001") is None


def test_items_jsonl_written_with_qids(runner, manifest):
    runner.run_one("R0001")
    items_file = rec.ids.items_path("R0001", runner.results_root)
    lines = items_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == manifest.by_id("R0001").n_items
    assert '"qid"' in lines[0]


def test_run_ready_only_runs_h2_rows(runner, manifest):
    records = runner.run_ready()
    n_h2 = sum(1 for r in manifest.runs if r.gate == "H2")
    assert len(records) == n_h2
    assert all(r.status == "Done" for r in records)


def test_integrity_demotes_short_run(runner, manifest):
    def short(row, cfg):
        # Intended the full benchmark but delivered one fewer item.
        return ExecResult(
            n_items=row.n_items - 1, metrics={"stub": True}, items=[], expected_n_items=row.n_items
        )

    record = runner.run_one("R0002", executor=short)
    assert record.status == "Failed"
    assert "integrity" in (record.error or "")


def test_parquet_rollup_and_duckdb_view(runner):
    runner.run_one("R0001")
    runner.run_one("R0002")
    parquet = runner.rollup()
    assert parquet is not None and parquet.exists()
    from mgr.tracking import store

    status_rows = store.v_status(parquet)
    total = sum(r["n"] for r in status_rows)
    assert total == 2
