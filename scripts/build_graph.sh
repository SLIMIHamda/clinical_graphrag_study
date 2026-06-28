#!/usr/bin/env bash
# build_graph.sh — one-time grounded-graph build (Step 2 -> gate G3).
# Runs on the pod with Neo4j up and NIM reachable. Keep this on on-demand (not
# spot) so a reclaim never loses the build (Doc 00 section 6).
#
# This wraps a project build entrypoint; the chunking/extraction parameters live
# in configs. After it completes and the coverage curve looks right, freeze
# graph_hash and set gates.G3.satisfied: true in configs/gates.yaml.
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

: "${NIM_API_KEY:?need NIM_API_KEY for entity extraction}"
: "${NEO4J_PASSWORD:?need NEO4J_PASSWORD}"

echo "[build_graph] starting Neo4j (if not running)…"
docker compose -f infra/neo4j/docker-compose.yml up -d

echo "[build_graph] building grounded graph from corpus chunks…"
# Entrypoint reads CORPUS chunks, extracts entities (NIM), links to UMLS, writes
# to Neo4j, and prints the coverage curve + graph_hash.
python -m mgr.graph.build_cli \
  --corpus "${CORPUS:?set CORPUS}" \
  --umls "${UMLS_DICT:?set UMLS_DICT (term->CUI json)}" \
  --neo4j-uri "bolt://localhost:7687" \
  --out "${RESULTS_ROOT:-/vol/results}/graph_build.json"

echo "[build_graph] done — review coverage, then set gates.G3.satisfied: true"
