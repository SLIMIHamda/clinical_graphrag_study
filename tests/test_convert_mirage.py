import json

from mgr.data.convert_mirage import convert_file, convert_records
from mgr.data.loader import load_items


def test_convert_records_mcq_and_yesno():
    mcq = {"q1": {"question": "What?", "options": {"A": "a", "B": "b"}, "answer": "B"}}
    rows = convert_records(mcq)
    assert rows[0]["qid"] == "q1"
    assert rows[0]["options"]["B"] == "b"
    assert rows[0]["answer"] == "B"

    yn = {"p1": {"question": "Is it?", "answer": "yes"}}
    rows = convert_records(yn)
    assert "options" not in rows[0]  # yes/no has no options
    assert rows[0]["answer"] == "yes"


def test_convert_file_writes_loadable_jsonl(tmp_path):
    mirage = {
        "mmlu": {"m1": {"question": "Q1", "options": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer": "C"}},
        "pubmedqa": {"p1": {"question": "P1", "answer": "maybe"}},
        "unknown_dataset": {"u1": {"question": "?", "answer": "z"}},  # ignored
    }
    src = tmp_path / "benchmark.json"
    src.write_text(json.dumps(mirage), encoding="utf-8")

    out = tmp_path / "data"
    written = convert_file(src, out)
    assert written == {"MMLU-Med": 1, "PubMedQA": 1}  # unknown dataset skipped

    # round-trips through the real loader with the right answer types
    mmlu = load_items("MMLU-Med", out, benchmark_type="MCQ")
    assert mmlu[0].answer_type == "mcq" and mmlu[0].gold == "C"
    pubmed = load_items("PubMedQA", out, benchmark_type="yes/no/maybe")
    assert pubmed[0].answer_type == "yes_no_maybe" and pubmed[0].gold == "maybe"
