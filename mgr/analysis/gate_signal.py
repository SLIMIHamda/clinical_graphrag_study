"""Phase-A signal analysis for the Gate-A selective-retrieval policy.

The oracle (mgr.analysis.rescue) says a *perfect* when-to-retrieve gate has real
headroom. Phase A asks the prerequisite question before any gate is trained:

    Do pre-retrieval uncertainty signals actually predict the retrieval reward?

For every item we have the base model's No-RAG outcome, the chosen RAG arm's
outcome, and the pre-retrieval feature vector x_q (mgr.gate.features). Define the
per-item retrieval reward

    r(q) = 1[RAG correct] - 1[No-RAG correct]  in {-1, 0, +1}

(+1 rescue, -1 break, 0 no change) and the rescueable label y = 1[r == +1].
A gate is any policy that maps x_q -> retrieve / don't; we score *any* policy by
routing each item to its RAG outcome when it retrieves and its No-RAG outcome
otherwise:

    routed_acc = mean( RAG_i if retrieve_i else NoRAG_i )
    recovered  = (routed_acc - NoRAG_acc) / (Oracle_acc - NoRAG_acc)

`recovered` is the headline: how much of the oracle ceiling a *realizable* gate
banks. This module reports (a) univariate AUROC of each signal vs y, (b) simple
threshold policies, (c) learned logistic / MLP rescue gates via cross-validation,
and (d) the four-outcome net-benefit gate -- each with accuracy, retrieval rate
and recovered fraction.

The net-benefit gate (paper Prop. 1, policy class iv) models breaks as well as
rescues: a multinomial logistic model over {rescue, break, both right, both
wrong} retrieves when P(rescue|x) - P(break|x) > lambda. Comparing it with the
rescue-only classifiers isolates what modelling breaks is worth.

IMPORTANT (see docs/gate_a_phase_a.md): on MedQA the positive class is tiny
(~tens of items), so learned-gate numbers here are a *plumbing dry-run*, not a
result -- every output carries an `underpowered` flag when n_rescueable is small.
The real train/validation/test evaluation happens at MIMIC scale.

Core (`gate_signal_analysis`) is pure numpy; the learned gates use scikit-learn
and degrade gracefully to a note if it is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mgr.gate.features import SCALAR_FEATURES

# Which direction of each signal means "the model is unsure -> retrieve". Used to
# orient univariate AUROC so >0.5 always reads as "uncertainty predicts rescue",
# and to pick the retrieve side of each threshold policy. +1: high => retrieve.
_ORIENT: dict[str, int] = {
    "entropy": +1,
    "confidence": -1,
    "margin": -1,
    "valid_mass": -1,
    "sc_entropy": +1,
    "sc_agreement": -1,
    "sc_matches_greedy": -1,
    "q_len_chars": +1,
    "q_len_words": +1,
    "n_options": +1,
}
# Features swept as standalone threshold gates (the interpretable baselines).
_THRESHOLD_FEATURES = ("entropy", "confidence", "margin", "valid_mass", "sc_entropy")

_DEFAULT_MIN_POS = 30  # below this many rescues (or breaks, for net-benefit), learned gates are dry-run only.

# Per-item retrieval outcome (paper Sec. 3): both wrong S0, both right S1,
# rescue S+ (No-RAG wrong, RAG right), break S- (No-RAG right, RAG wrong).
_S0, _S1, _RESCUE, _BREAK = 0, 1, 2, 3


def _outcomes(no_rag: np.ndarray, rag: np.ndarray) -> np.ndarray:
    return np.where(no_rag == rag, no_rag, 2 + no_rag).astype(int)


# --------------------------------------------------------------------------- #
# metric helpers (pure numpy)
# --------------------------------------------------------------------------- #
def _rank_average(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (like scipy.stats.rankdata 'average')."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank statistic; NaN if a class is empty or scores degenerate."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if not np.isfinite(scores).all():
        return float("nan")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _policy_scores(
    retrieve: np.ndarray,
    no_rag: np.ndarray,
    rag: np.ndarray,
    *,
    lam: float,
    no_rag_acc: float,
    oracle_acc: float,
) -> dict[str, Any]:
    retrieve = np.asarray(retrieve, dtype=bool)
    outcome = np.where(retrieve, rag, no_rag)
    acc = float(outcome.mean())
    rate = float(retrieve.mean())
    denom = oracle_acc - no_rag_acc
    recovered = round((acc - no_rag_acc) / denom, 4) if denom > 1e-9 else None
    return {
        "accuracy": round(acc, 4),
        "retrieval_rate": round(rate, 4),
        "objective": round(acc - lam * rate, 4),
        "recovered_fraction": recovered,
    }


def _best_threshold(
    values: np.ndarray,
    orient: int,
    no_rag: np.ndarray,
    rag: np.ndarray,
    *,
    lam: float,
    no_rag_acc: float,
    oracle_acc: float,
) -> dict[str, Any]:
    """Sweep a single-feature threshold gate; keep the one maximizing objective.

    ``orient == +1`` retrieves when ``value > tau`` (high = unsure); ``-1``
    retrieves when ``value < tau``. Endpoints include always/never retrieve.
    """
    v = np.asarray(values, dtype=float)
    uniq = np.unique(v[np.isfinite(v)])
    best: dict[str, Any] | None = None
    if orient >= 0:
        taus = np.concatenate(([-np.inf], uniq))  # -inf => retrieve all
        masks = ((v > t) for t in taus)
    else:
        taus = np.concatenate((uniq, [np.inf]))    # +inf => retrieve all
        masks = ((v < t) for t in taus)
    for tau, mask in zip(taus, masks):
        s = _policy_scores(mask, no_rag, rag, lam=lam, no_rag_acc=no_rag_acc, oracle_acc=oracle_acc)
        if best is None or s["objective"] > best["objective"]:
            best = {"threshold": (None if not np.isfinite(tau) else round(float(tau), 4)),
                    "retrieve_when": ("high" if orient >= 0 else "low"), **s}
    return best  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# feature matrix
# --------------------------------------------------------------------------- #
def _matrix(records: Sequence[Mapping[str, Any]], *, use_embedding: bool) -> tuple[np.ndarray, list[str]]:
    cols = list(SCALAR_FEATURES)
    scal = np.array([[float(r.get(c, 0.0)) for c in cols] for r in records], dtype=float)
    if use_embedding and records and isinstance(records[0].get("q_emb"), (list, tuple)):
        emb = np.array([r.get("q_emb", []) for r in records], dtype=float)
        if emb.ndim == 2 and emb.shape[1] > 0:
            scal = np.hstack([scal, emb])
            cols = cols + [f"emb{i}" for i in range(emb.shape[1])]
    return scal, cols


def _objective_tau(proba: np.ndarray, no_rag: np.ndarray, rag: np.ndarray, lam: float) -> float:
    """The retrieve-when-proba>tau threshold maximizing accuracy - lam*rate on this set."""
    best_tau, best_obj = -np.inf, -np.inf
    for tau in np.concatenate(([-np.inf], np.unique(proba))):
        ret = proba > tau
        obj = float(np.where(ret, rag, no_rag).mean()) - lam * float(ret.mean())
        if obj > best_obj:
            best_obj, best_tau = obj, float(tau)
    return best_tau


def _cv_learned_gate(
    X: np.ndarray,
    y: np.ndarray,
    no_rag: np.ndarray,
    rag: np.ndarray,
    *,
    kind: str,
    lam: float,
    no_rag_acc: float,
    oracle_acc: float,
    n_splits: int,
    seed: int,
) -> dict[str, Any] | None:
    """Cross-validated rescue classifier with **nested** threshold selection.

    Within each fold the model is fit on the training portion and the decision
    threshold tau is tuned on that same training portion; both are then applied
    to the held-out fold. So the routed accuracy / recovered fraction come from
    genuinely held-out decisions -- no threshold leakage (the trap that lets an
    MLP on noise embeddings fabricate recovery). ``oof_auroc`` is threshold-free.

    None if sklearn is missing or there are too few positives to stratify.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    k = min(n_splits, n_pos, n_neg)
    if k < 2:
        return None

    def _make():
        if kind == "logistic":
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(class_weight="balanced", max_iter=1000))
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(32, 8), max_iter=2000,
                                           random_state=seed))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    proba = np.full(len(y), np.nan, dtype=float)   # held-out proba (threshold-free metric)
    retrieve = np.zeros(len(y), dtype=bool)         # held-out decisions (nested tau)
    taus: list[float] = []
    for tr, te in skf.split(X, y):
        model = _make()
        model.fit(X[tr], y[tr])
        classes = list(model.classes_)
        idx1 = classes.index(1) if 1 in classes else None
        p_tr = model.predict_proba(X[tr])[:, idx1] if idx1 is not None else np.zeros(len(tr))
        p_te = model.predict_proba(X[te])[:, idx1] if idx1 is not None else np.zeros(len(te))
        tau = _objective_tau(p_tr, no_rag[tr], rag[tr], lam)  # tuned on TRAIN only
        proba[te] = p_te
        retrieve[te] = p_te > tau
        if np.isfinite(tau):
            taus.append(round(tau, 4))
    scores = _policy_scores(retrieve, no_rag, rag, lam=lam, no_rag_acc=no_rag_acc, oracle_acc=oracle_acc)
    scores["oof_auroc"] = round(_auroc(proba, y), 4)
    scores["cv_folds"] = k
    scores["mean_tau"] = round(float(np.mean(taus)), 4) if taus else None
    return scores


def _learned_gate(
    X: np.ndarray,
    y: np.ndarray,
    no_rag: np.ndarray,
    rag: np.ndarray,
    *,
    kind: str,
    lam: float,
    no_rag_acc: float,
    oracle_acc: float,
    n_splits: int,
    seed: int,
    n_seeds: int,
) -> dict[str, Any] | None:
    """Average the nested-CV gate over ``n_seeds`` CV shuffles so a single lucky
    split can't manufacture (or hide) recovery. Reports mean and std of the
    metrics that matter."""
    runs = [
        _cv_learned_gate(X, y, no_rag, rag, kind=kind, lam=lam, no_rag_acc=no_rag_acc,
                         oracle_acc=oracle_acc, n_splits=n_splits, seed=seed + s)
        for s in range(max(1, n_seeds))
    ]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    return _mean_over_seeds(runs, "mean over n_seeds nested-CV shuffles; tau tuned per-fold on training data only")


def _mean_over_seeds(runs: Sequence[Mapping[str, Any]], note: str) -> dict[str, Any]:
    def _agg(key: str) -> tuple[float | None, float | None]:
        vals = [r[key] for r in runs if r.get(key) is not None]
        if not vals:
            return None, None
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    acc, acc_sd = _agg("accuracy")
    rec, rec_sd = _agg("recovered_fraction")
    auc, auc_sd = _agg("oof_auroc")
    rate, _ = _agg("retrieval_rate")
    out = {
        "accuracy": acc, "accuracy_std": acc_sd,
        "retrieval_rate": rate,
        "recovered_fraction": rec, "recovered_fraction_std": rec_sd,
        "oof_auroc": auc, "oof_auroc_std": auc_sd,
        "cv_folds": runs[0]["cv_folds"], "n_seeds": len(runs),
        "note": note,
    }
    if any("oof_auroc_break" in r for r in runs):
        out["oof_auroc_break"], out["oof_auroc_break_std"] = _agg("oof_auroc_break")
    return out


def _cv_net_benefit_gate(
    X: np.ndarray,
    outcome: np.ndarray,
    no_rag: np.ndarray,
    rag: np.ndarray,
    *,
    lam: float,
    no_rag_acc: float,
    oracle_acc: float,
    n_splits: int,
    seed: int,
) -> dict[str, dict[str, Any]] | None:
    """Cross-validated four-outcome gate (paper Prop. 1, policy class iv).

    Per fold, a multinomial logistic model is fit on the training portion and
    scores the held-out items with ``net = P(rescue|x) - P(break|x)``. Two
    decision rules use the same fits:

    - ``plug_in``: retrieve when ``net > lam``, the optimal rule with the fitted
      probabilities plugged in; nothing is tuned. Class weights are left
      unbalanced so the outputs estimate the outcome rates the rule needs.
    - ``tuned``: the cut on ``net`` maximizing the objective on the training
      portion, the same nested procedure as the rescue classifiers.

    ``oof_auroc`` is ``net`` vs the rescue label (comparable to the rescue
    classifiers); ``oof_auroc_break`` is P(break|x) vs the break label.
    None if sklearn is missing or rescues/breaks are too few to stratify.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None
    is_rescue = (outcome == _RESCUE).astype(int)
    is_break = (outcome == _BREAK).astype(int)
    k = min(n_splits, int(is_rescue.sum()), int(is_break.sum()))
    if k < 2:
        return None

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    net = np.full(len(outcome), np.nan, dtype=float)
    p_break = np.full(len(outcome), np.nan, dtype=float)
    plug = np.zeros(len(outcome), dtype=bool)
    tuned = np.zeros(len(outcome), dtype=bool)
    for tr, te in skf.split(X, outcome):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[tr], outcome[tr])
        classes = list(model.classes_)

        def _scores(Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            P = model.predict_proba(Xs)
            col = lambda c: P[:, classes.index(c)] if c in classes else np.zeros(len(Xs))  # noqa: E731
            return col(_RESCUE) - col(_BREAK), col(_BREAK)

        net_tr, _ = _scores(X[tr])
        net_te, pb_te = _scores(X[te])
        tau = _objective_tau(net_tr, no_rag[tr], rag[tr], lam)  # tuned on TRAIN only
        net[te], p_break[te] = net_te, pb_te
        plug[te] = net_te > lam
        tuned[te] = net_te > tau

    common = {"oof_auroc": round(_auroc(net, is_rescue), 4),
              "oof_auroc_break": round(_auroc(p_break, is_break), 4),
              "cv_folds": k}

    def _scored(mask: np.ndarray) -> dict[str, Any]:
        return {**_policy_scores(mask, no_rag, rag, lam=lam, no_rag_acc=no_rag_acc,
                                 oracle_acc=oracle_acc), **common}

    return {"plug_in": _scored(plug), "tuned": _scored(tuned)}


def _net_benefit_gate(
    X: np.ndarray,
    outcome: np.ndarray,
    no_rag: np.ndarray,
    rag: np.ndarray,
    *,
    lam: float,
    no_rag_acc: float,
    oracle_acc: float,
    n_splits: int,
    seed: int,
    n_seeds: int,
) -> dict[str, dict[str, Any]] | None:
    """:func:`_cv_net_benefit_gate` averaged over ``n_seeds`` CV shuffles, per rule."""
    runs = [
        _cv_net_benefit_gate(X, outcome, no_rag, rag, lam=lam, no_rag_acc=no_rag_acc,
                             oracle_acc=oracle_acc, n_splits=n_splits, seed=seed + s)
        for s in range(max(1, n_seeds))
    ]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    return {
        "net_benefit": _mean_over_seeds(
            [r["plug_in"] for r in runs],
            "mean over n_seeds CV shuffles; retrieve when P(rescue)-P(break) > lambda, nothing tuned"),
        "net_benefit_tuned": _mean_over_seeds(
            [r["tuned"] for r in runs],
            "mean over n_seeds CV shuffles; cut on P(rescue)-P(break) tuned per-fold on training data only"),
    }


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def gate_signal_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    lam: float = 0.0,
    min_pos: int = _DEFAULT_MIN_POS,
    n_splits: int = 5,
    seed: int = 0,
    n_seeds: int = 5,
    use_embedding: bool = True,
) -> dict[str, Any]:
    """Phase-A signal analysis over joined per-item records.

    Each record needs ``no_rag_correct`` and ``rag_correct`` (0/1) plus the
    scalar features in :data:`mgr.gate.features.SCALAR_FEATURES`; ``q_emb`` is
    optional. Returns a JSON-serializable report; empty ``records`` -> {}.
    """
    records = [r for r in records if r.get("no_rag_correct") is not None and r.get("rag_correct") is not None]
    if not records:
        return {}
    no_rag = np.array([int(bool(r["no_rag_correct"])) for r in records], dtype=int)
    rag = np.array([int(bool(r["rag_correct"])) for r in records], dtype=int)
    n = len(records)
    y = ((rag == 1) & (no_rag == 0)).astype(int)  # rescueable
    n_pos = int(y.sum())

    no_rag_acc = float(no_rag.mean())
    rag_acc = float(rag.mean())
    oracle_acc = float(np.maximum(no_rag, rag).mean())  # keep NoRAG when right, else RAG
    breaks = int(((no_rag == 1) & (rag == 0)).sum())

    def _scored(mask: np.ndarray) -> dict[str, Any]:
        return _policy_scores(mask, no_rag, rag, lam=lam, no_rag_acc=no_rag_acc, oracle_acc=oracle_acc)

    # (a) univariate signal: AUROC of each oriented feature vs the rescueable label
    univariate = []
    for feat in SCALAR_FEATURES:
        vals = np.array([float(r.get(feat, 0.0)) for r in records], dtype=float)
        oriented = _ORIENT.get(feat, 1) * vals
        auc = _auroc(oriented, y)
        univariate.append({
            "feature": feat,
            "auroc": (None if np.isnan(auc) else round(auc, 4)),
            "retrieve_when": ("high" if _ORIENT.get(feat, 1) >= 0 else "low"),
        })
    univariate.sort(key=lambda d: (-(abs((d["auroc"] or 0.5) - 0.5))))

    # (b) threshold policies (+ always/never anchors)
    thresholds = {
        "always_retrieve": _scored(np.ones(n, dtype=bool)),
        "never_retrieve": _scored(np.zeros(n, dtype=bool)),
    }
    for feat in _THRESHOLD_FEATURES:
        vals = np.array([float(r.get(feat, 0.0)) for r in records], dtype=float)
        thresholds[feat] = _best_threshold(
            vals, _ORIENT.get(feat, 1), no_rag, rag,
            lam=lam, no_rag_acc=no_rag_acc, oracle_acc=oracle_acc,
        )

    # (c) learned gates (cross-validated); scalar-only and (optionally) +embedding
    X_scalar, _ = _matrix(records, use_embedding=False)
    learned: dict[str, Any] = {}
    for kind in ("logistic", "mlp"):
        res = _learned_gate(X_scalar, y, no_rag, rag, kind=kind, lam=lam,
                            no_rag_acc=no_rag_acc, oracle_acc=oracle_acc,
                            n_splits=n_splits, seed=seed, n_seeds=n_seeds)
        learned[kind] = res if res is not None else {"note": "skipped (sklearn missing or too few positives)"}
    if use_embedding:
        X_emb, _cols = _matrix(records, use_embedding=True)
        if X_emb.shape[1] > X_scalar.shape[1]:
            res = _learned_gate(X_emb, y, no_rag, rag, kind="mlp", lam=lam,
                                no_rag_acc=no_rag_acc, oracle_acc=oracle_acc,
                                n_splits=n_splits, seed=seed, n_seeds=n_seeds)
            learned["mlp_with_embedding"] = res if res is not None else {"note": "skipped"}
    nb = _net_benefit_gate(X_scalar, _outcomes(no_rag, rag), no_rag, rag, lam=lam,
                           no_rag_acc=no_rag_acc, oracle_acc=oracle_acc,
                           n_splits=n_splits, seed=seed, n_seeds=n_seeds)
    if nb is None:
        nb = {name: {"note": "skipped (sklearn missing or too few rescues/breaks)"}
              for name in ("net_benefit", "net_benefit_tuned")}
    learned.update(nb)

    underpowered = n_pos < min_pos
    underpowered_breaks = breaks < min_pos
    notes = [
        "threshold_policies are the best IN-SAMPLE operating point of a single signal "
        "(optimistic upper bound on that signal); learned_gates use nested CV and are held-out.",
    ]
    if underpowered:
        notes.append(
            f"UNDERPOWERED: only {n_pos} rescueable items (< {min_pos}). Learned-gate and "
            "recovered-fraction numbers are a plumbing dry-run, not a result -- the real "
            "train/val/test evaluation belongs at MIMIC scale."
        )
    if underpowered_breaks:
        notes.append(
            f"UNDERPOWERED (breaks): only {breaks} break items (< {min_pos}); the net-benefit "
            "gate's P(break|x) is not estimable enough to be a result."
        )
    if len({r.get("label_scoring") for r in records}) > 1:
        notes.append("MIXED label_scoring methods across records; option-score features are not comparable.")
    if oracle_acc - no_rag_acc <= 1e-9:
        notes.append("No oracle headroom (Oracle_acc == NoRAG_acc); recovered_fraction is undefined.")

    return {
        "n_items": n,
        "n_rescueable": n_pos,
        "n_breaks": breaks,
        "positive_rate": round(n_pos / n, 4),
        "lambda": lam,
        "underpowered": underpowered,
        "underpowered_breaks": underpowered_breaks,
        "baseline": {
            "no_rag_acc": round(no_rag_acc, 4),
            "rag_always_acc": round(rag_acc, 4),
            "oracle_acc": round(oracle_acc, 4),
            "oracle_gain": round(oracle_acc - no_rag_acc, 4),
        },
        "univariate_auroc": univariate,
        "threshold_policies": thresholds,
        "learned_gates": learned,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# loading / CLI
# --------------------------------------------------------------------------- #
def _rag_correct_by_qid(run_dir: str | Path, arm: str) -> dict[str, int]:
    """Pool an arm's per-item outcomes over seeds -> {qid: majority em}."""
    from mgr.analysis.rescue import _load_arm_items

    by_cond = _load_arm_items(Path(run_dir))
    arm_items = by_cond.get(arm, {})
    agg: dict[str, list[int]] = {}
    for (_seed, qid), em in arm_items.items():
        agg.setdefault(qid, []).append(int(em))
    return {qid: int(sum(v) / len(v) >= 0.5) for qid, v in agg.items()}


def load_records(
    features_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    arm: str | None = None,
) -> list[dict[str, Any]]:
    """Read the No-RAG feature rows (jsonl) and attach ``rag_correct``.

    ``rag_correct`` is taken from a row if present, else filled by joining on qid
    against ``arm``'s pooled outcomes in ``run_dir``. ``no_rag_correct`` accepts
    ``no_rag_correct`` / ``em`` / ``correct`` on the row.
    """
    rows = []
    for line in Path(features_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    rag_map = _rag_correct_by_qid(run_dir, arm) if (run_dir and arm) else {}
    out = []
    for r in rows:
        if r.get("gate_error"):
            continue  # the probe failed on this item, so its features are missing
        qid = str(r.get("qid")) if r.get("qid") is not None else None
        no_rag = r.get("no_rag_correct", r.get("em", r.get("correct")))
        rag = r.get("rag_correct")
        if rag is None and qid is not None:
            rag = rag_map.get(qid)
        if no_rag is None or rag is None:
            continue
        out.append({**r, "no_rag_correct": int(bool(no_rag)), "rag_correct": int(bool(rag))})
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m mgr.analysis.gate_signal <features.jsonl> "
              "[--run-dir DIR --arm NAME] [--lambda L] [--min-pos N]")
        return 0
    features_path = argv[0]
    def _opt(name: str, default: str | None = None) -> str | None:
        return argv[argv.index(name) + 1] if name in argv else default

    run_dir = _opt("--run-dir")
    arm = _opt("--arm")
    lam = float(_opt("--lambda", "0.0"))  # type: ignore[arg-type]
    min_pos = int(_opt("--min-pos", str(_DEFAULT_MIN_POS)))  # type: ignore[arg-type]

    records = load_records(features_path, run_dir=run_dir, arm=arm)
    if not records:
        print(f"no records: need no_rag_correct + rag_correct per item in {features_path}"
              + (" (or --run-dir/--arm to join)" if not (run_dir and arm) else ""))
        return 1
    report = gate_signal_analysis(records, lam=lam, min_pos=min_pos)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
