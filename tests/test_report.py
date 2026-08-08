import base64
import json

from mgr import report


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _full_run(root, *, with_figures=True):
    a = root / "artifacts"
    _write(a / "poc_summary.json", {
        "benchmark": "MMLU-Med", "n_items": 64, "gen_model_recorded": ["meta/llama-3.1-8b-instruct"],
        "gates": {"H2": True, "G3": True, "P3": False},
    })
    (a / "arm_metrics.csv").write_text(
        "condition,accuracy,coverage,item_errors,status\n"
        "No-RAG,0.75,1.0,0,Done\nHybrid-CARRF,0.797,1.0,0,Done\n",
        encoding="utf-8",
    )
    _write(a / "gate_ledger.json", {"H2": True, "G3": True, "P3": False})
    _write(a / "graph_report.json", {"n_chunks": 800, "n_concepts": 781, "linked_frac": 0.245,
                                     "coverage": {"exact": 0.225}})
    _write(a / "care_oracle.json", {"positive_rate": 0.0, "acc_with_rerank": 0.719, "P3": False})
    _write(a / "ragas.json", {"faithfulness": 0.178, "answer_relevance": 0.575})
    pr = root / "results" / "arms" / "per-run"
    _write(pr / "R0001.json", {
        "run_id": "R0001", "condition": "No-RAG", "backbone": "Llama-70B",
        "gen_model": "meta/llama-3.1-8b-instruct", "seed": 42, "status": "Done",
        "tokens": {"total": 10602}, "metrics": {"generation": {"accuracy": 0.75, "n_item_errors": 0}},
    })
    if with_figures:
        f = root / "figures"
        f.mkdir(parents=True, exist_ok=True)
        # a minimal 1x1 PNG is enough to exercise base64 embedding
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        (f / "F5_coverage.png").write_bytes(png)


def test_full_report_has_every_section_and_numbers(tmp_path):
    _full_run(tmp_path)
    md_path, html_path = report.build_report(tmp_path)
    md = md_path.read_text(encoding="utf-8")
    htmls = html_path.read_text(encoding="utf-8")

    # results present, no interpretation text
    for token in ("MMLU-Med", "Hybrid-CARRF", "0.797", "781", "0.178", "R0001"):
        assert token in md, token
        assert token in htmls, token
    # provenance: actual model shown, not just the manifest backbone
    assert "meta/llama-3.1-8b-instruct" in md
    # figure embedded as a data-URI in the self-contained HTML
    assert "data:image/png;base64," in htmls
    # no narrative fluff leaked in
    assert "interpret" not in md.lower() and "conclusion" not in md.lower()


def test_partial_run_marks_missing_not_crashes(tmp_path):
    # only the graph + gates got written before the run stopped (like the real
    # 2026-08-07 partial run)
    _write(tmp_path / "artifacts" / "gate_ledger.json", {"H2": True, "G3": True, "P3": False})
    _write(tmp_path / "artifacts" / "graph_report.json", {"n_concepts": 781})

    md_path, html_path = report.build_report(tmp_path)
    md = md_path.read_text(encoding="utf-8")

    assert "781" in md                       # what exists is reported
    assert md.count(report.MISSING) >= 3     # arms, oracle, ragas, per-run, figures absent
    assert "## Arms" in md and report.MISSING in md.split("## Arms", 1)[1]


def test_empty_run_dir_is_all_missing(tmp_path):
    md_path, _ = report.build_report(tmp_path)
    md = md_path.read_text(encoding="utf-8")
    assert "# Results" in md
    # every section degrades to (not produced) rather than raising
    assert md.count(report.MISSING) >= 6
