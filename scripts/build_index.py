#!/usr/bin/env python
"""Embed a corpus once into the artifacts the dense retriever reads.

Writes ``corpus_emb.npy`` (n x dim, L2-normalized) and ``corpus_ids.txt``, which
are what ``DENSE_EMB`` / ``DENSE_IDS`` point at. Embedding is a build step, not a
run step: the matrix is reused by every dense, hybrid and CA-RRF arm across every
seed, so this runs once per corpus.

Why this exists as a local script: the hosted embeddings tier allows ~40
requests/minute, which puts a 126k-chunk corpus somewhere around seven weeks.
The 600-chunk cap the first POC ran under was that limit showing through, and it
bounds how much retrieval quality that run could ever have demonstrated. On a
local GPU the same job is minutes, so corpus size stops being a design constraint.

    python scripts/build_index.py --corpus data/corpus.jsonl --out indices/

Passages are embedded with MedCPT's *article* encoder here; queries are embedded
at retrieval time with its *query* encoder (mgr.sweep._build_embedder). Using one
checkpoint for both degrades retrieval silently, so they are kept apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def load_corpus(path: str | Path, limit: int | None = None) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def build_embedder(args):
    if args.backend == "nim":
        import os

        from mgr.clients.nim import NimClient
        from mgr.clients.nim_adapters import NimEmbedder

        key = os.environ.get("NIM_API_KEY")
        if not key:
            raise SystemExit("--backend nim needs NIM_API_KEY")
        client = NimClient(base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com"), api_key=key)
        return NimEmbedder(client, model=args.model or "nvidia/nv-embedqa-e5-v5")

    from mgr.clients.local_embed import MEDCPT_ARTICLE, LocalEmbedder

    return LocalEmbedder(
        model_id=args.model or MEDCPT_ARTICLE,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Embed a corpus -> corpus_emb.npy + corpus_ids.txt")
    ap.add_argument("--corpus", required=True, help="chunks JSONL: {id, text, ...}")
    ap.add_argument("--out", default="indices", help="output directory")
    ap.add_argument("--backend", choices=["local", "nim"], default="local")
    ap.add_argument("--model", default=None, help="encoder checkpoint (default: MedCPT article encoder)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N chunks (smoke)")
    args = ap.parse_args(argv)

    import numpy as np

    records = load_corpus(args.corpus, args.limit)
    if not records:
        print(f"no records in {args.corpus}", file=sys.stderr)
        return 1

    ids = [str(r["id"]) for r in records]
    texts = [str(r.get("text", "")) for r in records]
    embedder = build_embedder(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    blocks, done = [], 0
    step = max(args.batch_size, 256)
    for i in range(0, len(texts), step):
        blocks.append(np.asarray(embedder(texts[i : i + step]), dtype=np.float32))
        done += len(blocks[-1])
        rate = done / max(1e-9, time.time() - t0)
        print(f"\r  embedded {done}/{len(texts)} ({rate:,.0f}/s)", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    matrix = np.vstack(blocks)
    if matrix.shape[0] != len(ids):
        print(f"embedder returned {matrix.shape[0]} vectors for {len(ids)} chunks", file=sys.stderr)
        return 1

    emb_path = out_dir / "corpus_emb.npy"
    ids_path = out_dir / "corpus_ids.txt"
    np.save(emb_path, matrix)
    ids_path.write_text("\n".join(ids), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"  {matrix.shape[0]:,} x {matrix.shape[1]} -> {emb_path}  ({elapsed:,.1f}s)")
    print(f"  {len(ids):,} ids      -> {ids_path}")
    print(f"\nexport DENSE_EMB={emb_path}\nexport DENSE_IDS={ids_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
