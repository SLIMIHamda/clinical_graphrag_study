"""B0 reference: the shipped values, and the guards around using them.

A placeholder that loads silently is worse than no gate — it converts "we never
checked our baselines against the literature" into a passing cross-check. So is
an absolute comparison against a different backbone, which is why the shipped
file carries its provenance and the report refuses to pass across a mismatch.
"""

import json
from pathlib import Path

import pytest

from mgr.eval.b0_crosscheck import (
    UnverifiedReference,
    crosscheck,
    delta_crosscheck,
    load_reference,
    load_reference_set,
)

REFERENCE = Path("configs/b0_reference.json")


def test_shipped_reference_is_verified_and_cited():
    raw = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert raw["_verified"] is True
    assert "2402.13178" in raw["_source"]
    assert raw["_backbone"] == "GPT-3.5"
    assert raw["_corpus"] == "Textbooks"


def test_shipped_reference_loads_with_every_value_filled():
    ref = load_reference(REFERENCE)
    assert ref["MedQA-US|No-RAG"] == 0.6504
    assert ref["MedQA-US|BM25"] == 0.6654
    assert all(isinstance(v, float) and 0.0 < v < 1.0 for v in ref.values())
    assert len(ref) == 10  # 5 benchmarks x {No-RAG, BM25}


def test_reference_set_carries_provenance():
    rs = load_reference_set(REFERENCE)
    assert rs.backbone == "GPT-3.5"
    assert rs.corpus == "Textbooks"
    assert rs.values["MMLU-Med|No-RAG"] == 0.7291


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


def test_unverified_file_raises(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"_verified": False, "MedQA-US|No-RAG": 0.65}), encoding="utf-8")
    with pytest.raises(UnverifiedReference, match="not verified"):
        load_reference(p)


def test_crosscheck_flags_an_out_of_tolerance_baseline():
    report = crosscheck(
        observed={"MedQA-US|No-RAG": 0.594}, reference={"MedQA-US|No-RAG": 0.6504}, tolerance=0.05
    )
    assert report.passed is False
    assert report.rows[0].delta == pytest.approx(-0.0564, abs=1e-9)


# -- backbone guard ----------------------------------------------------------

def test_absolute_check_cannot_pass_across_a_backbone_mismatch():
    """Our runs are Llama-3.1-8B; the reference is GPT-3.5. Landing inside the
    tolerance by luck must not be reported as a validated harness."""
    report = crosscheck(
        observed={"MedQA-US|No-RAG": 0.6504}, reference={"MedQA-US|No-RAG": 0.6504},
        observed_backbone="Llama-3.1-8B", reference_backbone="GPT-3.5",
    )
    assert report.rows[0].within_tol is True
    assert report.backbone_mismatch is True
    assert report.passed is False
    assert "not comparable" in report.summary()


def test_matching_backbone_passes_normally():
    report = crosscheck(
        observed={"MedQA-US|No-RAG": 0.66}, reference={"MedQA-US|No-RAG": 0.6504},
        observed_backbone="GPT-3.5", reference_backbone="GPT-3.5",
    )
    assert report.backbone_mismatch is False
    assert report.passed is True


def test_no_backbone_metadata_keeps_the_old_behaviour():
    report = crosscheck(observed={"MedQA-US|No-RAG": 0.66}, reference={"MedQA-US|No-RAG": 0.6504})
    assert report.passed is True


# -- delta check (the one that survives a backbone mismatch) -----------------

def test_delta_check_catches_retrieval_moving_the_wrong_way():
    """The POC's actual result: published BM25 gains +1.5pts on MedQA-US, ours
    lost 3.4. Same sign is the harness gate; magnitude is the finding."""
    ref = load_reference(REFERENCE)
    observed = {"MedQA-US|No-RAG": 0.594, "MedQA-US|BM25": 0.560}
    report = delta_crosscheck(observed, ref)

    row = next(r for r in report.rows if r.benchmark == "MedQA-US")
    assert row.observed_delta == pytest.approx(-0.034, abs=1e-9)
    assert row.reference_delta == pytest.approx(0.0150, abs=1e-9)
    assert row.sign_agrees is False
    assert report.passed is False


def test_delta_check_passes_when_the_direction_matches():
    ref = load_reference(REFERENCE)
    observed = {"MedQA-US|No-RAG": 0.50, "MedQA-US|BM25": 0.53}  # weaker, but helps
    report = delta_crosscheck(observed, ref)
    assert report.passed is True
    assert report.rows[0].sign_agrees is True


def test_delta_check_tolerates_a_benchmark_where_retrieval_hurts_in_the_paper():
    """MedMCQA/PubMedQA/BioASQ BM25-over-Textbooks are below no-retrieval in the
    source table, so a negative observed delta there is agreement, not failure."""
    ref = load_reference(REFERENCE)
    observed = {"PubMedQA|No-RAG": 0.40, "PubMedQA|BM25": 0.33}
    assert delta_crosscheck(observed, ref).passed is True


def test_delta_check_skips_benchmarks_without_both_arms():
    ref = load_reference(REFERENCE)
    report = delta_crosscheck({"MedQA-US|BM25": 0.56}, ref)  # no baseline observed
    assert report.rows == []
    assert report.passed is False  # nothing checked is not a pass
