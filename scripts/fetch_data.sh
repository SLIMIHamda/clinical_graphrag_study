#!/usr/bin/env bash
# fetch_data.sh — download everything heavy ONTO THE NETWORK VOLUME (/vol), once.
# Run on the pod (or any host with /vol mounted). Nothing large ever touches your
# laptop. Re-runnable: skips what already exists.
#
# Almost all of this is public. The clinical set (MIMIC) is NOT fetched here —
# it stays synthetic until the R1 governance gate (Doc 00 D4).
set -euo pipefail
[ -f .env ] && set -a && . ./.env && set +a

VOL="${VOL:-/vol}"
DATA="${VOL}/data/mirage"
IDX="${VOL}/indices"
mkdir -p "$DATA" "$IDX/bm25" "$IDX/dense"

# 1) MIRAGE QA (small) — questions + answers, then convert to our JSONL.
if [ ! -f "$DATA/benchmark.json" ]; then
  echo "[fetch] MIRAGE benchmark.json"
  curl -L -o "$DATA/benchmark.json" \
    https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json
fi
python -m mgr.data.convert_mirage --mirage "$DATA/benchmark.json" --out-dir "$DATA"

# 2) MedRAG corpora (large). Prefer StatPearls+Textbooks first (small, high-signal
#    for USMLE-style MedQA); add wikipedia/pubmed only if a benchmark needs them.
#    These are on HuggingFace as MedRAG/{statpearls,textbooks,wikipedia,pubmed}.
CORPORA="${CORPORA:-statpearls textbooks}"
for c in $CORPORA; do
  if [ ! -d "$VOL/corpus/$c" ]; then
    echo "[fetch] MedRAG corpus: $c"
    huggingface-cli download "MedRAG/$c" --repo-type dataset --local-dir "$VOL/corpus/$c"
  fi
done

# 3) Precomputed MedCPT embeddings / FAISS, if available for your chunking.
#    If not, embed once via NIM (free tier) and persist to $IDX/dense — the
#    DenseIndex.from_embeddings path then costs nothing per run.
echo "[fetch] (optional) place precomputed MedCPT embeddings under $IDX/dense"

echo "[fetch] done. Volume layout:"
echo "  $DATA/*.jsonl   (QA)"
echo "  $VOL/corpus/*   (corpora)"
echo "  $IDX/{bm25,dense}  (indices)"
