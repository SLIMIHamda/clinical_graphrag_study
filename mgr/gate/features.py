"""Pre-retrieval uncertainty features for the Gate-A selective-retrieval policy.

Gate A routes each question to *retrieve* or *don't retrieve* using only signals
available before retrieval — i.e. from the base model answering with no context.
This module turns those raw signals (option logprobs, self-consistency samples,
the question string, an optional question embedding) into the scalar feature
vector x_q the gate consumes.

Deliberately stdlib-only (``math`` + ``collections``): it runs *inside* the
generation loop on the serving box and must not drag numpy/sklearn or the
numpy<2 scienv into the runner. The learning/scoring lives in
``mgr.analysis.gate_signal`` (numpy/sklearn, analysis env).

Reward framing (docs/gate_a_phase_a.md): the training target is the retrieval
reward ``r(q) = 1[RAG correct] - 1[No-RAG correct]`` in {-1,0,+1}; this module
only produces the *inputs* x_q, never the label.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

# The scalar features emitted here, in a fixed order. The question embedding is
# carried separately as ``q_emb`` (a vector, not a scalar) because only the
# learned gates consume it; the univariate signal analysis ignores it. Kept as a
# module constant so the runner, the analysis layer and the instrumentation spec
# all agree on one schema.
SCALAR_FEATURES: tuple[str, ...] = (
    "confidence",         # max_c P(option c)      -- high => model is sure
    "entropy",            # H(P over options)      -- high => unsure  => retrieve
    "margin",             # P(top1) - P(top2)      -- low  => unsure  => retrieve
    "sc_agreement",       # modal share over k samples (self-consistency)
    "sc_entropy",         # H over the k sampled answers
    "sc_matches_greedy",  # share of samples equal to the greedy answer
    "q_len_chars",
    "q_len_words",
    "n_options",
)

_LOGPROB_FLOOR = -50.0  # exp(-50) ~ 2e-22: an option absent from top-k is ~0, not missing.


def _normalize(probs: Mapping[str, float]) -> dict[str, float]:
    """Clamp negatives to 0 and renormalize to sum 1; uniform if all-zero."""
    clamped = {k: max(0.0, float(v)) for k, v in probs.items()}
    z = sum(clamped.values())
    if z <= 0.0:
        n = len(clamped) or 1
        return {k: 1.0 / n for k in clamped}
    return {k: v / z for k, v in clamped.items()}


def option_probs_from_logprobs(
    top_logprobs: Sequence[Mapping[str, Any]],
    option_letters: Sequence[str],
) -> dict[str, float]:
    """Fold an OpenAI-style ``top_logprobs`` list (for the answer-token position)
    into a probability over the option letters.

    Each entry is ``{"token": ..., "logprob": ...}``. Tokens are matched to an
    option by their stripped, upper-cased first character, so ``" B"`` / ``"B"``
    / ``"b"`` all count toward option B. Options never seen in the top-k keep a
    tiny floor so the distribution stays full-support before renormalization.
    """
    letters = [c.strip().upper()[:1] for c in option_letters]
    mass = {c: 0.0 for c in letters if c}
    for entry in top_logprobs or ():
        tok = str(entry.get("token", "")).strip().upper()
        if not tok:
            continue
        c = tok[0]
        if c in mass:
            mass[c] += math.exp(float(entry.get("logprob", _LOGPROB_FLOOR)))
    floor = math.exp(_LOGPROB_FLOOR)
    mass = {c: (v if v > 0.0 else floor) for c, v in mass.items()}
    return _normalize(mass)


def answer_top_logprobs(
    content: Sequence[Mapping[str, Any]],
    option_letters: Sequence[str],
) -> list[dict[str, Any]]:
    """From an OpenAI logprobs ``content`` list (per-token dicts), return the
    ``top_logprobs`` of the first generated token that *is* an option letter — so
    a chatty "The answer is B" still resolves to B's alternatives. Falls back to
    the first token's alternatives, then to an empty list.
    """
    letters = {c.strip().upper()[:1] for c in option_letters if c}
    for tok in content or ():
        t = str(tok.get("token", "")).strip().upper()
        if t and t[0] in letters:
            return list(tok.get("top_logprobs") or [])
    if content:
        return list(content[0].get("top_logprobs") or [])
    return []


def confidence_features(option_probs: Mapping[str, float]) -> dict[str, float]:
    """confidence (top-1 prob), entropy, and top1-top2 margin over the options."""
    p = _normalize(dict(option_probs))
    ordered = sorted(p.values(), reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    entropy = -sum(v * math.log(v) for v in p.values() if v > 0.0)
    return {"confidence": top1, "entropy": entropy, "margin": top1 - top2}


def self_consistency_features(
    samples: Sequence[str], greedy_answer: str
) -> dict[str, float]:
    """Stability of the answer across ``k`` stochastic samples.

    ``sc_agreement`` is the modal answer's share (1.0 = perfectly stable),
    ``sc_entropy`` the entropy of the sampled-answer distribution, and
    ``sc_matches_greedy`` the share of samples equal to the greedy answer. All
    zero when no samples were drawn (self-consistency disabled).
    """
    normed = [s for s in (str(x).strip().upper() for x in samples) if s]
    if not normed:
        return {"sc_agreement": 0.0, "sc_entropy": 0.0, "sc_matches_greedy": 0.0}
    counts = Counter(normed)
    k = len(normed)
    _, modal_n = counts.most_common(1)[0]
    probs = [c / k for c in counts.values()]
    entropy = -sum(pp * math.log(pp) for pp in probs if pp > 0.0)
    greedy = str(greedy_answer).strip().upper()
    matches_greedy = sum(1 for s in normed if s == greedy) / k if greedy else 0.0
    return {
        "sc_agreement": modal_n / k,
        "sc_entropy": entropy,
        "sc_matches_greedy": matches_greedy,
    }


def structural_features(question: str, n_options: int) -> dict[str, float]:
    q = question or ""
    return {
        "q_len_chars": float(len(q)),
        "q_len_words": float(len(q.split())),
        "n_options": float(n_options),
    }


def compute_features(
    *,
    option_probs: Mapping[str, float],
    greedy_answer: str,
    question: str,
    n_options: int,
    samples: Sequence[str] | None = None,
    q_emb: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Assemble the full Gate-A feature record for one item.

    Returns every key in :data:`SCALAR_FEATURES`, plus ``q_emb`` (a list) when a
    question embedding is supplied. All inputs are pre-retrieval, so the record
    is exactly what the deployed gate would see at decision time.
    """
    feats: dict[str, Any] = {}
    feats.update(confidence_features(option_probs))
    feats.update(self_consistency_features(samples or [], greedy_answer))
    feats.update(structural_features(question, n_options))
    if q_emb is not None:
        feats["q_emb"] = [float(x) for x in q_emb]
    return feats
