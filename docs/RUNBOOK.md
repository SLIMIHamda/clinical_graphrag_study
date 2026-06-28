# Runbook — from clean repo to the 70B sweep

Ordered checklist. Steps 0–2 cost nothing and need no GPU. The GPU pod only
comes up at Step 4, and `with_pod.sh` guarantees it tears down afterward.

## 0. One-time accounts & secrets (local, free)
- [ ] NVIDIA NIM key from **build.nvidia.com** (free tier; embeddings/reranker/
      judge/graph-extraction only — never generation).
- [ ] RunPod account + a **Network Volume** (note its `VOLUME_ID`).
- [ ] `cp .env.example .env` and fill in `NIM_API_KEY`, `RUNPOD_API_KEY`,
      `VOLUME_ID`, `NEO4J_PASSWORD`. **Never commit `.env`.**
- [ ] `pip install -e .[dev]` and `pytest` → all green. (On the pod add `[serve]`
      for neo4j + vllm.)

## 1. Get the QA data (local, ~tens of MB)
- [ ] Download MIRAGE `benchmark.json`, then:
      `python -m mgr.data.convert_mirage --mirage benchmark.json --out-dir data`
- [ ] Confirm `data/MMLU-Med.jsonl` etc. exist. (Corpora stay off your laptop.)

## 2. Freeze the manifest contract (local, free)
- [ ] `python -m manifest.lock` → validates the workbook, writes
      `manifest/manifest.lock.json`. Budget must read $591 / $887.

## 3. Stage the heavy data on the volume (cloud disk, ~free)
- [ ] On a **cheap CPU pod** (or the GPU pod), with `/vol` mounted:
      `bash scripts/fetch_data.sh` → corpora + (precomputed) embeddings land on
      `/vol`. Prefer StatPearls+Textbooks first to keep the volume small.

## 4. Smoke → gate H2 (first GPU use, minutes)
- [ ] `bash infra/runpod/with_pod.sh bash scripts/smoke.sh`
      (provisions A100 → serves vLLM → runs No-RAG + BM25 on 200 items → tears
      down). Expect `PASS`.
- [ ] Set `gates.H2.satisfied: true` in `configs/gates.yaml` (evidence = the
      smoke report). This unblocks the 118 baseline rows.

## 5. Graph build → gate G3 (one-time, on-demand)
- [ ] Build the grounded chunk-level graph + UMLS grounding; freeze `graph_hash`.
- [ ] Set `gates.G3.satisfied: true`. Unblocks graph + all hybrid arms.

## 6. The sweep (the paid step)
- [ ] Start the idle guard on a cheap host:
      `python infra/guards/pod_watch.py --idle-threshold-s 1800`
- [ ] `bash infra/runpod/with_pod.sh bash scripts/run_sweep.sh`
      (saturating async submit over Ready rows; resume skips Done; budget gate
      halts if projected actual > ceiling).
- [ ] Watch `cost_actual_usd` (GPU-hours) vs the ~$280–320 target.

## 7. Oracle + CARe → gate P3, then stats & figures
- [ ] Compute oracle rerank labels from B3+M1; set `gates.P3.satisfied: true`.
- [ ] Run the CARe arm; then the stats pipeline (audit → CIs → exact-p → Holm →
      effect sizes) and emit figures/tables.

## Safety net (always on during GPU use)
1. `with_pod.sh` trap tears down on exit/crash/Ctrl-C.
2. `down.sh` is idempotent.
3. `pod_watch.py` kills a pod left idle past the threshold.
