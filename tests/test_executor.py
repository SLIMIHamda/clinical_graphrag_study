import json

import pytest

from manifest.manifest import load_manifest
from mgr.data.loader import load_items, write_items_fixture
from mgr.generate.executor import RAGExecutor
from mgr.runner import Runner
from mgr.tracking import record as rec


class FakeGenClient:
    """Returns a fixed normalized answer; reports deterministic token usage."""

    def __init__(self, answer="B"):
        self.answer = answer
        self.calls = 0

    def complete_text(self, model, messages, **params):
        self.calls += 1
        return self.answer, {"in": 100, "out": 1}


@pytest.fixture
def fixture_data(tmp_path):
    rows = [
        {"qid": f"mmlu_{i:03d}", "question": f"Q{i}?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
         "answer": "B" if i % 2 == 0 else "C"}
        for i in range(10)
    ]
    write_items_fixture("MMLU-Med", tmp_path, rows)
    return tmp_path


def test_loader_reads_fixture(fixture_data):
    items = load_items("MMLU-Med", fixture_data, benchmark_type="MCQ")
    assert len(items) == 10
    assert items[0].answer_type == "mcq"
    assert items[0].options["A"] == "a"


def test_norag_executor_through_runner(fixture_data, tmp_path):
    m = load_manifest()
    # R0001 = No-RAG / MMLU-Med / Llama-70B / s42, gated on H2.
    client = FakeGenClient(answer="B")
    execu = RAGExecutor(client=client, data_root=fixture_data, n_items=10)
    runner = Runner(
        manifest=m,
        gate_ledger={"H2": True, "G3": False, "P3": False},
        results_root=tmp_path / "results",
    )
    record = runner.run_one("R0001", executor=execu)

    assert record is not None
    assert record.status == "Done"
    assert record.executor == "real"
    assert record.n_items == 10
    assert client.calls == 10
    # Half the golds are "B" -> accuracy 0.5 with a constant "B" prediction.
    assert record.metrics["generation"]["accuracy"] == pytest.approx(0.5)
    assert record.metrics["generation"]["coverage"] == 1.0
    assert record.tokens["total"] == 10 * 101
    assert record.cost_actual_usd > 0  # real tokens -> real (token-side) cost

    # per-item JSONL is qid-keyed and complete
    lines = rec.ids.items_path("R0001", runner.results_root).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    first = json.loads(lines[0])
    assert first["qid"] == "mmlu_000"
    assert first["answer_norm"] == "B"
    assert {"em", "f1", "gold", "latency_s", "tokens"} <= set(first)
