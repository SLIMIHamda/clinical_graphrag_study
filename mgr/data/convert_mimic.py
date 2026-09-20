"""Convert a credentialed MIMIC-derived MCQ set into our benchmark JSONL + corpus.

Companion to :mod:`mgr.data.convert_mirage`. MIRAGE ships one fixed schema, so
that converter hard-codes it; the MIMIC-side MCQ sets (EHRNoteQA and friends)
each carry their own field names and are chosen per project, so the mapping is
supplied on the command line instead of guessed.

Two artifacts come out, because a chart benchmark is not just questions:

* ``{benchmark}.jsonl`` -- the normalized MCQ rows the loader reads.
* ``corpus.jsonl`` -- ``{"id", "text", "patient_id", ...}`` chunks of the patient
  notes. **The answer must live in this corpus rather than in the model**; that
  is what makes the task retrieval-mandatory, and what the No-RAG floor in the
  pilot go/no-go actually tests.

``patient_id`` is carried on both sides on purpose. It is what makes
*patient-scoped* retrieval (search only this patient's chart) a filter over an
existing index rather than a rebuild, so that scoping decision stays reversible
after the data is converted.

This module never reaches the network and never calls a model -- it is pure
reshaping, so it runs on the box that holds the credentialed data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from .loader import write_items_fixture

DEFAULT_BENCHMARK = "MIMIC-QA"
LETTERS = "ABCDEFGH"


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, or a JSON list / dict-of-records, into a list of dicts.

    Whole-document JSON is tried first; a multi-line JSONL file fails that parse
    and falls through to the per-line reader. (MIRAGE ships a dict keyed by qid,
    the MIMIC-side sets tend to ship JSONL, and a one-line JSONL file is
    indistinguishable from a single JSON record — so it is read as one.)
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return [{"_key": k, **v} for k, v in obj.items()]
        return [obj]
    raise ValueError(f"{path}: expected JSONL, a JSON list, or a dict of records")


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Fixed-width character chunks with overlap, whitespace-collapsed."""
    text = " ".join((text or "").split())
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def normalize_options(raw: Any) -> dict[str, str] | None:
    """Accept ``{"A": ...}``, ``["opt", ...]``, or None -> lettered dict."""
    if isinstance(raw, dict) and raw:
        return {(str(k).strip().upper()[:1] or str(k)): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)) and raw:
        return {LETTERS[i]: str(v) for i, v in enumerate(raw) if i < len(LETTERS)}
    return None


def normalize_answer(raw: Any, options: dict[str, str] | None) -> str | None:
    """Return the option *letter*.

    Handles a letter already ("B"), a 0- or 1-based index, and the answer text
    spelled out -- the three shapes these sets ship in. Anything else is returned
    as-is rather than guessed at, so a bad mapping surfaces as a mismatch in the
    run instead of a silently wrong gold label.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if options:
        if s.upper() in options:
            return s.upper()
        if s.isdigit():
            i, keys = int(s), list(options)
            if 0 <= i < len(keys):
                return keys[i]
            if 1 <= i <= len(keys):
                return keys[i - 1]
        for k, v in options.items():
            if str(v).strip() == s:
                return k
    return s


def convert_records(
    records: Iterable[dict[str, Any]],
    *,
    qid_key: str = "qid",
    question_key: str = "question",
    options_key: str = "options",
    answer_key: str = "answer",
    patient_key: str | None = "patient_id",
) -> list[dict[str, Any]]:
    """Map raw records to the normalized MCQ rows the loader reads."""
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        options = normalize_options(r.get(options_key))
        row: dict[str, Any] = {
            "qid": str(r.get(qid_key, r.get("_key", i))),
            "question": str(r.get(question_key, "")).strip(),
            "answer": normalize_answer(r.get(answer_key), options),
        }
        if options:
            row["options"] = options
        if patient_key and r.get(patient_key) is not None:
            row["patient_id"] = str(r[patient_key])  # -> BenchmarkItem.meta["patient_id"]
        rows.append(row)
    return rows


def build_corpus(
    records: Iterable[dict[str, Any]],
    *,
    note_key: str = "note",
    patient_key: str | None = "patient_id",
    notes_by_id: dict[str, str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> Iterator[dict[str, Any]]:
    """Yield ``{"id", "text", "patient_id", "chunk_index"}`` chunks, one pass per patient.

    Notes come from the record itself (``note_key``) or from a side file keyed by
    patient (``notes_by_id``). Many MCQ sets ask several questions about one
    chart, so a patient's notes are chunked once however many questions cite them.
    """
    seen: set[str] = set()
    for i, r in enumerate(records):
        pid = str(r.get(patient_key, i)) if patient_key else str(i)
        if pid in seen:
            continue
        seen.add(pid)
        note = r.get(note_key) if note_key in r else None
        if note is None and notes_by_id is not None:
            note = notes_by_id.get(pid)
        if isinstance(note, (list, tuple)):
            note = "\n".join(str(n) for n in note)
        for j, ch in enumerate(chunk_text(str(note or ""), chunk_size, chunk_overlap)):
            yield {"id": f"{pid}_c{j:04d}", "text": ch, "patient_id": pid, "chunk_index": j}


def write_corpus(chunks: Iterable[dict[str, Any]], out_path: str | Path) -> int:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MIMIC-derived MCQ set -> benchmark JSONL + note corpus")
    ap.add_argument("--src", required=True, help="source records (JSONL, or JSON list/dict)")
    ap.add_argument("--out-dir", default="data", help="output dir for {benchmark}.jsonl")
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--corpus-out", default=None, help="chunks JSONL (default: <out-dir>/corpus.jsonl)")
    ap.add_argument("--notes", default=None, help="optional side file of note records")
    ap.add_argument("--qid-key", default="qid")
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--options-key", default="options")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--patient-key", default="patient_id")
    ap.add_argument("--note-key", default="note")
    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    args = ap.parse_args(argv)

    records = read_records(args.src)
    if not records:
        print(f"no records read from {args.src}", file=sys.stderr)
        return 1

    notes_by_id = None
    if args.notes:
        notes_by_id = {
            str(n.get(args.patient_key, n.get("id", i))): str(n.get("text", n.get(args.note_key, "")))
            for i, n in enumerate(read_records(args.notes))
        }

    rows = convert_records(
        records, qid_key=args.qid_key, question_key=args.question_key,
        options_key=args.options_key, answer_key=args.answer_key, patient_key=args.patient_key,
    )
    path = write_items_fixture(args.benchmark, args.out_dir, rows)

    corpus_out = args.corpus_out or str(Path(args.out_dir) / "corpus.jsonl")
    n_chunks = write_corpus(
        build_corpus(
            records, note_key=args.note_key, patient_key=args.patient_key,
            notes_by_id=notes_by_id, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        ),
        corpus_out,
    )

    n_gold = sum(1 for r in rows if r.get("answer"))
    print(f"  {args.benchmark:<12} {len(rows):>6} items -> {path}")
    print(f"  corpus       {n_chunks:>6} chunks -> {corpus_out}")
    if n_gold != len(rows):
        print(f"WARNING: {len(rows) - n_gold} rows have no gold answer -- check --answer-key", file=sys.stderr)
    if n_chunks == 0:
        print("WARNING: empty corpus -- check --note-key/--notes; every RAG arm would be a no-op", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
