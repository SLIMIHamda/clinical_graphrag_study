"""CLI for the grounded-graph build (Step 2).

Reads corpus chunks (JSONL: {id, text, ...provenance}), extracts entities, grounds
them to UMLS CUIs from a term->CUI dictionary, writes the chunk/concept graph, and
emits a build report with the frozen graph_hash.

Both halves are selectable, because the original Neo4j + NIM combination is
unavailable on a machine holding credentialed chart text (no hosted calls) and on
any box without a database:

* ``--extractor ngram`` (default) is a dictionary scan — offline, free,
  deterministic. ``--extractor nim`` is the hosted LLM path.
* ``--store memory`` (default) writes a JSON snapshot that ``GRAPH_SNAPSHOT``
  points at. ``--store neo4j`` writes to the database.

Without a snapshot the in-memory graph dies with the process, which is why the
graph arms were previously reachable only through Neo4j.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .build import build_graph
from .umls import UMLSLinker


def _load_chunks(path: str | Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the grounded graph (-> gate G3)")
    ap.add_argument("--corpus", required=True, help="chunks JSONL: {id, text, ...provenance}")
    ap.add_argument("--umls", required=True, help="term->CUI JSON dictionary")
    ap.add_argument("--extractor", choices=["ngram", "nim"], default="ngram")
    ap.add_argument("--store", choices=["memory", "neo4j"], default="memory")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--out-graph", default="results/graph_snapshot.json",
                    help="JSON snapshot path (--store memory); point GRAPH_SNAPSHOT at it")
    ap.add_argument("--out", default="results/graph_build.json")
    ap.add_argument("--fuzzy", action="store_true", help="enable fuzzy linking (slow on a large dictionary)")
    ap.add_argument("--min-link-rate", type=float, default=0.0,
                    help="fail if fewer than this fraction of chunks ground to >=1 concept")
    args = ap.parse_args(argv)

    exact_map = json.loads(Path(args.umls).read_text(encoding="utf-8"))
    linker = UMLSLinker(exact_map, fuzzy_enabled=args.fuzzy)

    if args.extractor == "nim":
        from mgr.clients.nim import NimClient
        from mgr.clients.nim_adapters import NimEntityExtractor

        extractor = NimEntityExtractor(
            NimClient(base_url=os.environ["NIM_BASE_URL"], api_key=os.environ["NIM_API_KEY"])
        )
    else:
        from .extract import NgramExtractor

        extractor = NgramExtractor.from_linker_dict(exact_map)

    if args.store == "neo4j":
        from .neo4j_store import Neo4jStore

        store = Neo4jStore(uri=args.neo4j_uri, user="neo4j", password=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    else:
        from .store import InMemoryGraphStore

        store = InMemoryGraphStore()

    report = build_graph(_load_chunks(args.corpus), store, extractor, linker)
    out = {
        "n_chunks": report.n_chunks,
        "n_concepts": report.n_concepts,
        "n_links": report.n_links,
        "graph_hash": report.graph_hash,
        "extractor": args.extractor,
        "store": args.store,
    }

    if args.store == "memory":
        from .store import save_snapshot

        out["snapshot"] = args.out_graph
        out["graph_hash"] = save_snapshot(store, args.out_graph)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    # A graph that links almost nothing is the failure mode that made CA-RRF
    # identical to plain RRF last time (1 concept over 60 chunks), and it passes
    # every structural check. Surface it here rather than in the results.
    link_rate = (report.n_links / report.n_chunks) if report.n_chunks else 0.0
    if link_rate < args.min_link_rate:
        print(f"[build_graph] FAIL: {link_rate:.3f} links/chunk < --min-link-rate {args.min_link_rate}",
              file=sys.stderr)
        return 1
    if report.n_concepts <= 1:
        print(f"[build_graph] WARNING: only {report.n_concepts} distinct concept(s) — "
              "the concept layer is effectively empty and CA-RRF will equal plain RRF",
              file=sys.stderr)
    print(f"[build_graph] wrote {args.out}; set gates.G3 evidence to it and flip satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
