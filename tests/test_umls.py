import pytest

from mgr.graph.umls import UMLSLinker, coverage, normalize_term

EXACT = {
    "myocardial infarction": "C0027051",
    "aspirin": "C0004057",
    "diabetes mellitus": "C0011849",
    "hypertension": "C0020538",
}
ABBREV = {"MI": "myocardial infarction", "HTN": "hypertension", "DM": "diabetes mellitus"}


def test_normalize_term():
    assert normalize_term("Myocardial-Infarction!") == "myocardial infarction"


def test_exact_linking():
    lk = UMLSLinker(EXACT)
    r = lk.link_term("Aspirin")
    assert r.cui == "C0004057" and r.method == "exact"


def test_abbreviation_linking_resolves_to_expansion_cui():
    lk = UMLSLinker(EXACT, ABBREV)
    r = lk.link_term("MI")
    assert r.cui == "C0027051" and r.method == "abbrev"  # MI -> myocardial infarction CUI


def test_fuzzy_linking_catches_variants():
    lk = UMLSLinker(EXACT, fuzzy_threshold=0.85)
    r = lk.link_term("myocardial infarctions")  # plural / typo-ish variant
    assert r.cui == "C0027051" and r.method == "fuzzy"


def test_unlinked_when_nothing_matches():
    lk = UMLSLinker(EXACT, fuzzy_enabled=False)
    assert lk.link_term("xyzzy plague").method == "none"


def test_concepts_set_excludes_misses():
    lk = UMLSLinker(EXACT, ABBREV)
    cuis = lk.concepts(["aspirin", "MI", "totally unknown"])
    assert cuis == {"C0004057", "C0027051"}


def test_coverage_curve_improves_with_each_tier():
    # exact-only would miss the abbreviation and the variant.
    mentions = ["aspirin", "MI", "myocardial infarctions", "hypertension", "qwerty"]
    lk = UMLSLinker(EXACT, ABBREV, fuzzy_threshold=0.85)
    rep = coverage(mentions, lk)
    curve = rep.curve
    assert curve["exact"] < curve["exact+abbrev"] < curve["exact+abbrev+fuzzy"]
    assert rep.unlinked == 1               # only 'qwerty'
    assert curve["exact+abbrev+fuzzy"] == pytest.approx(0.8)  # 4 of 5 linked
