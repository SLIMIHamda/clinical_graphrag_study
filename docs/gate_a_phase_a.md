# Gate A — instrumentation spec & Phase-A plan

Gate A is a **pre-retrieval selective-retrieval policy**: for each question, decide
*whether to retrieve at all* using only signals available before retrieval (the base
model's No-RAG pass). The scientific contribution is **not** "we added a neural net" — it
is *quantifying how much of the oracle rescue ceiling a realizable, uncertainty-based
policy can recover*. The NN is merely the policy approximator.

This doc is the contract the runner must satisfy to produce the inputs, and the plan for
the analysis that consumes them. The analysis harness (`mgr.analysis.gate_signal`,
`mgr.gate.features`) is already built and unit-tested against synthetic data — it runs the
moment real features land.

---

## 0. Reward framing (the target)

For each question `q`, with the base model's No-RAG outcome and the chosen RAG arm's
outcome, define the **retrieval reward**

```
r(q) = 1[RAG correct] − 1[No-RAG correct]  ∈ {−1, 0, +1}
       +1 = rescue     −1 = break     0 = no change
```

and the rescueable label `y(q) = 1[r == +1]`. A gate routes each item to its RAG outcome
when it retrieves, else its No-RAG outcome. We score any gate by

```
routed_acc = mean( RAG_i if retrieve_i else NoRAG_i )
recovered  = (routed_acc − NoRAG_acc) / (Oracle_acc − NoRAG_acc)   ← the headline
```

The reward (not bare "rescueability") is the right frame: routing keeps No-RAG on the
items it already gets right, so a good gate **avoids breaks and banks rescues in one
objective**, tied directly to net accuracy. Cost enters as `max_θ [ accuracy − λ·retrieval_rate ]`.

> **Scope guard.** On MedQA the positive class is tiny (~tens of items), so *learned-gate*
> numbers are a **plumbing dry-run**, not a result. MedQA validates the pipeline and tests
> the *premise* (do signals correlate with reward?). The real train/val/test evaluation
> happens at **MIMIC** scale, where retrieval is mandatory and positives are plentiful.

---

## 1. What the runner must log — per item, No-RAG arm

Emit these as **flat top-level keys** on each per-item row (the analysis loader reads them
directly). Names must match `mgr.gate.features.SCALAR_FEATURES` exactly.

| key | type | source |
| --- | --- | --- |
| `qid` | str | existing |
| `no_rag_correct` | 0/1 | existing `em` on the No-RAG row |
| `confidence` | float | `max_c P(option c)` from answer-token logprobs |
| `entropy` | float | `H(P over options)` |
| `margin` | float | `P(top1) − P(top2)` |
| `sc_agreement` | float | modal share over `k` self-consistency samples |
| `sc_entropy` | float | entropy of the `k` sampled answers |
| `sc_matches_greedy` | float | share of samples equal to the greedy answer |
| `q_len_chars`, `q_len_words`, `n_options` | float | from the question/options |
| `q_emb` | list[float] | question embedding (optional but requested — see §2.4) |

Optional raw fields for audit (not required by the analysis): `option_logprobs`
(`{"A":.., "B":..}`), `self_consistency_samples` (`["A","A","B",...]`).

**Do not compute these by hand** — call `mgr.gate.features.compute_features(...)`, which
returns exactly these keys. It is stdlib-only, so it runs inside the generation loop.

The RAG outcome for the reward is **not** logged here — it is joined from the existing RAG
arms at analysis time (§3).

---

## 2. How to obtain each signal (serving stack)

All three sources are OpenAI-compatible and already reachable through the existing clients.

### 2.1 Answer-option logprobs → confidence / entropy / margin
The chat client forwards `**params` into the payload
([`openai_compat.py:137`](../mgr/clients/openai_compat.py)), so request logprobs by passing:

```python
resp = client.chat(model, messages, logprobs=True, top_logprobs=10, max_tokens=2, temperature=0)
content = resp["choices"][0]["logprobs"]["content"]   # list of per-token dicts
top = content[0]["top_logprobs"]                        # candidates for the answer token
```

Then:

```python
from mgr.gate.features import option_probs_from_logprobs
option_probs = option_probs_from_logprobs(top, option_letters=["A","B","C","D"])
```

`option_probs_from_logprobs` matches tokens by their stripped upper-cased first char (so
`" B"`/`"B"`/`"b"` all count for B) and floors unseen options. If the MCQ answer isn't the
*first* generated token for some prompt format, point it at the position that holds the
letter.

### 2.2 Self-consistency → sc_* (the one non-free feature)
Draw `k` (≈5) generations at `temperature≈0.7` and pass the answer letters:

```python
samples = [normalize(client.complete_text(model, msgs, temperature=0.7, max_tokens=2)[0], atype)
           for _ in range(k)]
```

**Budget this in the cost term.** `k` extra base passes to save one retrieval is a net loss
on MedQA (cheap retrieval) but plausibly a win on MIMIC (expensive long-chart retrieval).
Start with `k=5`; it can be dropped to 0 (features go to 0) if the free logprob signals
already separate.

### 2.3 Structural
`structural_features(question, n_options)` — free, from the strings.

### 2.4 Question embedding → q_emb
The chat API returns no hidden states, so embed the **question text** with the existing
embeddings endpoint (the retrieval encoder) — a pre-retrieval, one-call-per-item feature:

```python
vec = client.embeddings(embed_model, [question])["data"][0]["embedding"]
```

(`NimClient.embeddings`, [`nim.py:48`](../mgr/clients/nim.py).) Store as `q_emb`. Note: at
MedQA's ~25 positives an embedding-fed gate **will** overfit — the analysis flags this — so
`q_emb` matters for MIMIC; log it now so no re-run is needed later.

---

## 3. Which run, and the join

- **Re-run the No-RAG arm only** on the same MedQA-US items (N=256). No retrieval, plus the
  logprob request, `k` samples, and one embedding call per item — cheap (No-RAG was ~72k
  tokens/seed). One seed is enough for features; the reward's RAG side comes from the
  existing 3-seed run.
- **RAG arm for the reward** — a design choice. On MedQA the oracle is flat across arms, so
  it barely matters; use the **deployed pipeline** (`Hybrid-CARRF`, or `BM25` as the
  simplest strong arm). `gate_signal` joins it by qid via `--run-dir/--arm`.

### Code integration points — WIRED (in-repo)
The runner is already wired; normal runs are untouched (capture is off by default and only the
No-RAG `NullRetriever` arm acts on it):
- **Client** — `VLLMClient.complete_text_logprobs(...)` ([`vllm.py`](../mgr/clients/vllm.py))
  requests logprobs and returns the `content` list. Unit-tested against a fake transport.
- **Executor** — `RAGExecutor` gained a `gate: GateCapture` field
  ([`executor.py`](../mgr/generate/executor.py)); on the No-RAG arm it takes the logprob path,
  draws `n_samples` self-consistency completions, embeds the question, calls
  `compute_features`, and merges the flat feature keys (+ `gate_cost_tokens`, kept off the
  arm's token total) into each item row. A misconfig (capture on, client without
  `complete_text_logprobs`) raises **before** the arm runs.
- **Sweep** — `Resources.gate` ([`sweep.py`](../mgr/sweep.py)) is threaded into every arm;
  only No-RAG captures.

### Student run — exact steps (the only remaining code touch is the notebook client)
The POC generates via the notebook's `NimGenerationClient`, not `VLLMClient`, so add the same
method to it (NIM llama-3.1-8b supports logprobs on the OpenAI-compatible API). Drop this into
the `NimGenerationClient` cell — it needs no `_SAFE_GEN` change (logprobs are passed explicitly):

```python
    def complete_text_logprobs(self, model, messages, *, top_logprobs=10, **params):
        p = {k: v for k, v in params.items() if k in _SAFE_GEN}
        p.setdefault("temperature", self.temperature); p.setdefault("max_tokens", self.max_tokens)
        r = self._c.chat(self.model, messages, logprobs=True, top_logprobs=top_logprobs, **p)
        u = r.get("usage", {}); ch = r["choices"][0]
        content = (ch.get("logprobs") or {}).get("content") or []
        return (ch["message"]["content"],
                {"in": int(u.get("prompt_tokens", 0)), "out": int(u.get("completion_tokens", 0))},
                list(content))
```

Then enable capture on the No-RAG arm by passing a `GateCapture` into the sweep `Resources`:

```python
from mgr.generate.executor import GateCapture
resources = Resources(
    gen_client=GEN, data_root=..., retrievers=..., ...,
    gate=GateCapture(
        enabled=True, n_samples=5, sample_temperature=0.7, top_logprobs=10,
        embed_client=EMB, embed_model=CFG.EMBED_MODEL,   # EMB = the NIM embeddings client; drop these two for no q_emb
    ),
)
```

Re-running the sweep now writes the enriched No-RAG `items.jsonl` (every other arm is unchanged).
Only the No-RAG arm needs to re-run — RAG outcomes for the reward come from the existing run.

---

## 4. Phase-A analysis (already built)

Once the enriched No-RAG `items.jsonl` exists:

```bash
python -m mgr.analysis.gate_signal <features.jsonl> --run-dir results/poc_runs --arm Hybrid-CARRF --lambda 0.02
```

Reports, as JSON:
- **baseline** — No-RAG / RAG-always / **oracle** accuracy and the oracle gain;
- **univariate_auroc** — AUROC of each oriented signal vs the rescueable label (the premise
  test: does uncertainty predict rescue?);
- **threshold_policies** — always / never / entropy-τ / confidence-τ / margin-τ / sc_entropy-τ,
  each with accuracy, retrieval rate, recovered fraction (best **in-sample** operating point);
- **learned_gates** — logistic and MLP (scalars, and MLP+embedding), evaluated by
  **nested CV** (τ tuned per-fold on training data, scored on held-out folds) and averaged
  over seeds, with an `underpowered` flag when positives are scarce.

**How to read the premise test.** If free signals give AUROC ≈ 0.5, an MLP won't rescue it —
the routing problem has no learnable pre-retrieval signal on this data. If AUROC climbs to
~0.65–0.75, that justifies the trainable gate at MIMIC scale.

---

## 5. Sequence

1. **Instrumentation** — runner wired (§3). *Student adds the one notebook client method, enables `GateCapture`, and re-runs the No-RAG pass.*
2. **Oracle** — done (`mgr.analysis.rescue`, the ~0.646 ceiling).
3. **Signal analysis** — `mgr.analysis.gate_signal` (premise test on MedQA). ← unblocked by step 1
4. **Baselines** — always / never / entropy / confidence / logistic (in the same report).
5. **Tiny MLP (+ all features)** — reported now as a dry-run; a *result* only at MIMIC scale.
6. **Held-out eval** — train → val-tuned τ → frozen test. **MIMIC.**
7. **Add Gate B (CARe)** — report A, B, A+B on quality *and* retrieval cost. **MIMIC.**

Steps 5–7 as *findings* require MIMIC (retrieval-mandatory, large positive class). MedQA
carries steps 1–4 as a premise test and an end-to-end plumbing check.
