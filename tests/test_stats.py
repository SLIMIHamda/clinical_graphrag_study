import numpy as np
import pytest

from mgr.stats.bootstrap import bootstrap_ci, bootstrap_diff_ci
from mgr.stats.effect_size import cliffs_delta, cohens_d_paired, interpret_cliffs
from mgr.stats.holm import holm_bonferroni
from mgr.stats.io import align_by_qid
from mgr.stats.permutation import paired_permutation_test


# ----- bootstrap ----------------------------------------------------------- #

def test_bootstrap_ci_brackets_mean():
    vals = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    ci = bootstrap_ci(vals, n_boot=5000, seed=1)
    assert ci.lo <= ci.point <= ci.hi
    assert abs(ci.point - vals.mean()) < 1e-9


def test_bootstrap_diff_ci_positive_when_a_better():
    a = np.ones(50)
    b = np.zeros(50)
    ci = bootstrap_diff_ci(a, b, n_boot=2000, seed=2)
    assert ci.lo > 0  # a strictly better -> CI excludes 0


# ----- permutation --------------------------------------------------------- #

def test_permutation_no_effect_gives_high_p():
    rng = np.random.default_rng(0)
    x = rng.random(40)
    res = paired_permutation_test(x, x.copy(), n_perm=20000, seed=3)
    assert res.diff == 0.0
    assert res.p_value == pytest.approx(1.0, abs=1e-6)


def test_permutation_strong_effect_hits_floor_and_refuses():
    a = np.ones(60)
    b = np.zeros(60)
    res = paired_permutation_test(a, b, n_perm=10000, seed=4)
    assert res.at_floor                       # never exceeded -> bound, not a point
    assert res.p_value < 1.0 / res.n_perm * 2
    with pytest.raises(ValueError):
        res.assert_resolved()                 # refuses to report the floor as p


def test_permutation_exact_for_small_n():
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    res = paired_permutation_test(a, b, n_perm=100000, seed=5)
    assert res.exact
    assert 0.0 < res.p_value <= 1.0


# ----- holm ---------------------------------------------------------------- #

def test_holm_rejects_smallest_controls_family():
    res = holm_bonferroni({"c1": 0.001, "c2": 0.04, "c3": 0.20}, alpha=0.05)
    by = {r.label: r for r in res}
    assert by["c1"].reject              # 0.001 <= 0.05/3
    assert not by["c3"].reject          # 0.20 not significant
    assert by["c1"].p_adj <= by["c2"].p_adj <= by["c3"].p_adj  # monotone


# ----- effect size --------------------------------------------------------- #

def test_cohens_d_and_cliffs_delta():
    a = np.ones(30)
    b = np.zeros(30)
    assert cohens_d_paired(a, b) == 0.0         # zero variance in the difference
    assert cliffs_delta(a, b) == 1.0            # total dominance
    assert interpret_cliffs(1.0) == "large"
    assert interpret_cliffs(0.05) == "negligible"


# ----- io alignment -------------------------------------------------------- #

def test_align_by_qid_pairs_shared_items():
    a = [{"qid": "q1", "em": 1}, {"qid": "q2", "em": 0}]
    b = [{"qid": "q1", "em": 0}, {"qid": "q2", "em": 1}]
    va, vb, qids = align_by_qid(a, b, metric="em")
    assert qids == ["q1", "q2"]
    assert list(va) == [1.0, 0.0]
    assert list(vb) == [0.0, 1.0]


def test_align_by_qid_rejects_mismatched_sets():
    a = [{"qid": "q1", "em": 1}]
    b = [{"qid": "q1", "em": 1}, {"qid": "q2", "em": 1}]
    with pytest.raises(ValueError):
        align_by_qid(a, b, metric="em")
