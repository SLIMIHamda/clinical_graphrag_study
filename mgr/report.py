"""Results-only run report.

Reads a run directory's artifacts and emits a single Markdown file and a
self-contained HTML file (figures inlined as data-URIs) that hold nothing but
the results: metadata, the arm table, gates, graph/oracle/RAGAS numbers, the
CARe frontier, per-run provenance, and the figures. No prose, no interpretation.

Robust to partial runs: a section whose source artifact is absent is marked
``(not produced)`` rather than crashing, so an interrupted run still reports what
it reached.

Pure stdlib (no pandas/matplotlib) so it runs anywhere and is unit-testable.

CLI:  python -m mgr.report <run_dir> [--out <dir>]
"""

from __future__ import annotations

import base64
import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Artifact/figure names the notebook writes (kept in one place so a rename is a
# one-line change here).
ARTIFACTS = {
    "summary": "artifacts/poc_summary.json",
    "arms": "artifacts/arm_metrics.csv",
    "gates": "artifacts/gate_ledger.json",
    "graph": "artifacts/graph_report.json",
    "oracle": "artifacts/care_oracle.json",
    "ragas": "artifacts/ragas.json",
}
FIGURES = [
    ("F3 — Retrieval–Generation Decomposition (C1)", "figures/F3_rgd.png"),
    ("F4 — CARe cost–quality frontier (C3)", "figures/F4_pareto.png"),
    ("F5 — UMLS coverage curve", "figures/F5_coverage.png"),
]
PER_RUN_GLOB = "results/arms/per-run/R*.json"

MISSING = "(not produced)"


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    title: str
    # Either a list-of-dict "table", or a flat {k: v} "kv", or None when absent.
    table: list[dict[str, Any]] | None = None
    kv: dict[str, Any] | None = None
    present: bool = True


def _read_json(p: Path) -> Any | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_csv(p: Path) -> list[dict[str, str]] | None:
    try:
        with p.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return None


def _per_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(run_dir.glob(PER_RUN_GLOB)):
        rec = _read_json(p)
        if not isinstance(rec, dict):
            continue
        gen = rec.get("metrics", {}).get("generation", {})
        rows.append(
            {
                "run_id": rec.get("run_id"),
                "condition": rec.get("condition"),
                "backbone": rec.get("backbone"),
                "gen_model": rec.get("gen_model") or MISSING,
                "seed": rec.get("seed"),
                "status": rec.get("status"),
                "accuracy": gen.get("accuracy"),
                "n_item_errors": gen.get("n_item_errors"),
                "tokens_total": (rec.get("tokens") or {}).get("total"),
            }
        )
    return rows


def collect(run_dir: str | Path) -> list[Section]:
    """Read a run directory into ordered report sections."""
    run = Path(run_dir)
    a = {k: _read_json(run / v) for k, v in ARTIFACTS.items() if k != "arms"}
    arms = _read_csv(run / ARTIFACTS["arms"])
    sections: list[Section] = []

    # Run metadata (facts, not prose): what ran and under what gates.
    s = a.get("summary")
    if isinstance(s, dict):
        flat = {}
        for k, v in s.items():
            flat[k] = ", ".join(f"{kk}={vv}" for kk, vv in v.items()) if isinstance(v, dict) else v
        sections.append(Section("Run", kv=flat))
    else:
        sections.append(Section("Run", present=False))

    sections.append(Section("Arms", table=arms) if arms else Section("Arms", present=False))

    g = a.get("gates")
    sections.append(
        Section("Gates", kv=g) if isinstance(g, dict) else Section("Gates", present=False)
    )

    gr = a.get("graph")
    if isinstance(gr, dict):
        flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in gr.items()}
        sections.append(Section("Graph (G3)", kv=flat))
    else:
        sections.append(Section("Graph (G3)", present=False))

    o = a.get("oracle")
    sections.append(
        Section("CARe oracle (P3)", kv=o) if isinstance(o, dict) else Section("CARe oracle (P3)", present=False)
    )

    r = a.get("ragas")
    sections.append(
        Section("RAGAS grounding", kv=r) if isinstance(r, dict) else Section("RAGAS grounding", present=False)
    )

    per_run = _per_run_rows(run)
    sections.append(
        Section("Per-run records", table=per_run) if per_run else Section("Per-run records", present=False)
    )
    return sections


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _md_table(rows: list[dict[str, Any]]) -> str:
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _md_kv(kv: dict[str, Any]) -> str:
    return _md_table([{"key": k, "value": _fmt(v)} for k, v in kv.items()])


def render_markdown(sections: list[Section], *, figures_present: list[tuple[str, str]]) -> str:
    out: list[str] = ["# Results"]
    for sec in sections:
        out.append(f"\n## {sec.title}")
        if not sec.present:
            out.append(f"\n{MISSING}")
        elif sec.table is not None:
            out.append("\n" + _md_table(sec.table))
        elif sec.kv is not None:
            out.append("\n" + _md_kv(sec.kv))
    out.append("\n## Figures")
    if figures_present:
        for title, rel in figures_present:
            out.append(f"\n**{title}**\n\n![{title}]({rel})")
    else:
        out.append(f"\n{MISSING}")
    return "\n".join(out) + "\n"


_CSS = """
:root{color-scheme:light dark}
body{font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif;margin:2rem;max-width:1100px}
h1{font-size:1.5rem;margin:0 0 1rem}h2{font-size:1.05rem;margin:1.6rem 0 .5rem;border-bottom:1px solid #8884;padding-bottom:.2rem}
table{border-collapse:collapse;margin:.3rem 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #8886;padding:.3rem .55rem;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#8881}
.missing{color:#b00;font-style:italic}
img{max-width:100%;height:auto;border:1px solid #8884;margin:.3rem 0}
figure{margin:.6rem 0}figcaption{font-weight:600;margin-bottom:.3rem}
"""


def _html_table(rows: list[dict[str, Any]]) -> str:
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    thead = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(row.get(c)))}</td>" for c in cols) + "</tr>"
        for row in rows
    )
    return f"<table><thead>{thead}</thead><tbody>{body}</tbody></table>"


def _data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def render_html(sections: list[Section], *, figures_embedded: list[tuple[str, Path]]) -> str:
    parts: list[str] = [
        "<!doctype html><meta charset='utf-8'><title>Results</title>",
        f"<style>{_CSS}</style>",
        "<h1>Results</h1>",
    ]
    for sec in sections:
        parts.append(f"<h2>{html.escape(sec.title)}</h2>")
        if not sec.present:
            parts.append(f"<p class='missing'>{MISSING}</p>")
        elif sec.table is not None:
            parts.append(_html_table(sec.table))
        elif sec.kv is not None:
            parts.append(_html_table([{"key": k, "value": _fmt(v)} for k, v in sec.kv.items()]))
    parts.append("<h2>Figures</h2>")
    if figures_embedded:
        for title, p in figures_embedded:
            parts.append(
                f"<figure><figcaption>{html.escape(title)}</figcaption>"
                f"<img alt='{html.escape(title)}' src='{_data_uri(p)}'></figure>"
            )
    else:
        parts.append(f"<p class='missing'>{MISSING}</p>")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_report(run_dir: str | Path, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` for a run. Returns their paths."""
    run = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    sections = collect(run)
    figs_present = [(t, rel) for t, rel in FIGURES if (run / rel).exists()]
    figs_embedded = [(t, run / rel) for t, rel in FIGURES if (run / rel).exists()]

    md_path = out / "report.md"
    html_path = out / "report.html"
    md_path.write_text(render_markdown(sections, figures_present=figs_present), encoding="utf-8")
    html_path.write_text(render_html(sections, figures_embedded=figs_embedded), encoding="utf-8")
    return md_path, html_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m mgr.report <run_dir> [--out <dir>]")
        return 0
    run_dir = argv[0]
    out_dir = None
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
    md_path, html_path = build_report(run_dir, out_dir)
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
