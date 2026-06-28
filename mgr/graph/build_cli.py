"""CLI for the grounded-graph build (Step 2). Runtime: needs Neo4j + NIM.

Reads corpus chunks (JSONL: {id, text, ...provenance}), extracts entities via NIM,
grounds them to UMLS CUIs from a term->CUI dictionary, writes the chunk/concept
graph to Neo4j, and emits a build report with the frozen graph_hash.
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
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--out", default="results/graph_build.json")
    args = ap.parse_args(argv)

    from mgr.clients.nim import NimClient
    from mgr.clients.nim_adapters import NimEntityExtractor

    from .neo4j_store import Neo4jStore

    exact_map = json.loads(Path(args.umls).read_text(encoding="utf-8"))
    linker = UMLSLinker(exact_map)
    extractor = NimEntityExtractor(NimClient(base_url=os.environ["NIM_BASE_URL"], api_key=os.environ["NIM_API_KEY"]))
    store = Neo4jStore(uri=args.neo4j_uri, user="neo4j", password=os.environ.get("NEO4J_PASSWORD", "neo4j"))

    report = build_graph(_load_chunks(args.corpus), store, extractor, linker)
    out = {
        "n_chunks": report.n_chunks,
        "n_concepts": report.n_concepts,
        "n_links": report.n_links,
        "graph_hash": report.graph_hash,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[build_graph] wrote {args.out}; set gates.G3 evidence to it and flip satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
