"""The shipped B0 reference must not be usable until it is filled in.

A placeholder that loads silently is worse than no gate: it converts "we never
checked our baselines against the literature" into a passing cross-check, and
every accuracy claim downstream inherits that.
"""

import json
from pathlib import Path

import pytest

from mgr.eval.b0_crosscheck import UnverifiedReference, crosscheck, load_reference

REFERENCE = Path("configs/b0_reference.json")


def test_shipped_reference_is_marked_unverified():
    raw = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert raw["_verified"] is False


def test_loading_the_shipped_reference_raises(tmp_path):
    with pytest.raises(UnverifiedReference, match="not verified"):
        load_reference(REFERENCE)


def test_verified_but_unfilled_still_raises(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"_verified": True, "MedQA-US|No-RAG": 0.7, "MedQA-US|BM25": None}), encoding="utf-8")
    with pytest.raises(UnverifiedReference, match="unfilled"):
        load_reference(p)


def test_fully_filled_reference_loads(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(
        json.dumps({
            "_verified": True, "_source": "Xiong et al., Table 2",
            "MedQA-US|No-RAG": 0.70, "MedQA-US|BM25": 0.71,
        }),
        encoding="utf-8",
    )
    ref = load_reference(p)
    assert ref == {"MedQA-US|No-RAG": 0.70, "MedQA-US|BM25": 0.71}


def test_require_verified_false_is_the_test_escape_hatch(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"MedQA-US|No-RAG": 0.70}), encoding="utf-8")
    assert load_reference(p, require_verified=False) == {"MedQA-US|No-RAG": 0.70}


def test_crosscheck_flags_an_out_of_tolerance_baseline():
    report = crosscheck(
        observed={"MedQA-US|No-RAG": 0.594}, reference={"MedQA-US|No-RAG": 0.70}, tolerance=0.05
    )
    assert report.passed is False
    assert report.rows[0].delta == pytest.approx(-0.106, abs=1e-9)
