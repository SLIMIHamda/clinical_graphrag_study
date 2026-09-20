import json

from mgr.data.convert_mimic import (
    build_corpus,
    chunk_text,
    convert_records,
    main,
    normalize_answer,
    normalize_options,
    read_records,
)
from mgr.data.loader import load_items


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


# -- shape normalization ----------------------------------------------------

def test_options_from_dict_and_list():
    assert normalize_options({"A": "x", "B": "y"}) == {"A": "x", "B": "y"}
    assert normalize_options(["x", "y", "z"]) == {"A": "x", "B": "y", "C": "z"}
    assert normalize_options(None) is None
    assert normalize_options([]) is None


def test_answer_accepts_letter_index_and_text():
    opts = {"A": "10 mg", "B": "20 mg", "C": "5 mg"}
    assert normalize_answer("B", opts) == "B"
    assert normalize_answer("b", opts) == "B"
    assert normalize_answer(1, opts) == "B"       # 0-based
    assert normalize_answer("20 mg", opts) == "B"  # spelled out
    assert normalize_answer(None, opts) is None


def test_answer_is_passed_through_not_guessed_when_unmatchable():
    # a wrong --answer-key should surface as a gold mismatch in the run, not as a
    # plausible-looking letter invented here
    assert normalize_answer("Z", {"A": "x"}) == "Z"


# -- chunking ---------------------------------------------------------------

def test_chunks_overlap_and_collapse_whitespace():
    chunks = chunk_text("a  b\n\nc " * 200, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert "\n" not in "".join(chunks)


def test_empty_note_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text(None) == []


# -- conversion -------------------------------------------------------------

def test_convert_records_to_loader_schema():
    rows = convert_records([
        {"qid": "q1", "question": "dose?", "options": ["10 mg", "20 mg"], "answer": 0, "patient_id": "p1"},
    ])
    assert rows[0] == {
        "qid": "q1", "question": "dose?", "answer": "A",
        "options": {"A": "10 mg", "B": "20 mg"}, "patient_id": "p1",
    }


def test_custom_key_mapping():
    rows = convert_records(
        [{"id": "7", "stem": "what?", "choices": ["a", "b"], "gold": "b", "subject_id": "s9"}],
        qid_key="id", question_key="stem", options_key="choices",
        answer_key="gold", patient_key="subject_id",
    )
    assert rows[0]["qid"] == "7"
    assert rows[0]["answer"] == "B"
    assert rows[0]["patient_id"] == "s9"


def test_corpus_chunks_once_per_patient():
    records = [
        {"patient_id": "p1", "note": "x " * 500},
        {"patient_id": "p1", "note": "x " * 500},  # second question, same chart
        {"patient_id": "p2", "note": "y " * 500},
    ]
    chunks = list(build_corpus(records, chunk_size=200, chunk_overlap=0))
    assert {c["patient_id"] for c in chunks} == {"p1", "p2"}
    assert len({c["id"] for c in chunks}) == len(chunks)  # ids unique


def test_corpus_reads_notes_from_a_side_file():
    records = [{"patient_id": "p1", "question": "?"}]
    chunks = list(build_corpus(records, notes_by_id={"p1": "note body " * 50}, chunk_size=100))
    assert chunks and all(c["patient_id"] == "p1" for c in chunks)


# -- input formats ----------------------------------------------------------

def test_read_records_handles_jsonl_and_json_list(tmp_path):
    jl = write_jsonl(tmp_path / "a.jsonl", [{"qid": "1"}, {"qid": "2"}])
    assert len(read_records(jl)) == 2

    js = tmp_path / "b.json"
    js.write_text(json.dumps([{"qid": "1"}]), encoding="utf-8")
    assert len(read_records(js)) == 1


# -- CLI --------------------------------------------------------------------

def test_cli_writes_benchmark_and_corpus_that_the_loader_reads(tmp_path):
    src = write_jsonl(tmp_path / "src.jsonl", [
        {"qid": "q1", "question": "dose?", "options": ["10 mg", "20 mg"],
         "answer": "A", "patient_id": "p1", "note": "prescribed 10 mg " * 100},
        {"qid": "q2", "question": "route?", "options": ["PO", "IV"],
         "answer": "B", "patient_id": "p2", "note": "given IV " * 100},
    ])
    rc = main([
        "--src", str(src), "--out-dir", str(tmp_path / "data"),
        "--corpus-out", str(tmp_path / "data" / "corpus.jsonl"), "--benchmark", "MIMIC-QA",
    ])
    assert rc == 0

    items = load_items("MIMIC-QA", tmp_path / "data")
    assert [i.qid for i in items] == ["q1", "q2"]
    assert items[0].gold == "A"
    assert items[0].options == {"A": "10 mg", "B": "20 mg"}
    # patient_id survives into meta -> patient-scoped retrieval stays possible
    assert items[0].meta["patient_id"] == "p1"

    corpus = read_records(tmp_path / "data" / "corpus.jsonl")
    assert corpus and {c["patient_id"] for c in corpus} == {"p1", "p2"}
    assert all({"id", "text"} <= set(c) for c in corpus)


def test_cli_fails_loudly_on_an_empty_corpus(tmp_path):
    """A chart benchmark with no chart text makes every RAG arm a silent no-op."""
    src = write_jsonl(tmp_path / "src.jsonl", [{"qid": "q1", "question": "?", "options": ["a", "b"], "answer": "A"}])
    rc = main(["--src", str(src), "--out-dir", str(tmp_path / "data")])
    assert rc == 1
