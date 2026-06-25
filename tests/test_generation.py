from mgr.eval.answer_format_audit import audit
from mgr.generate import prompts
from mgr.generate.extract import normalize
from mgr.metrics.generation import exact_match, score, token_f1


# ----- extraction ---------------------------------------------------------- #

def test_normalize_mcq_variants():
    assert normalize("B", "mcq") == "B"
    assert normalize("The answer is C.", "mcq") == "C"
    assert normalize("(D)", "mcq") == "D"
    assert normalize("Answer: A", "mcq") == "A"
    assert normalize("I am not sure", "mcq") is None


def test_normalize_yesno():
    assert normalize("Yes, definitely.", "yes_no_maybe") == "yes"
    assert normalize("maybe", "yes_no_maybe") == "maybe"
    assert normalize("maybe", "yes_no") is None  # maybe invalid for yes/no
    assert normalize("No.", "yes_no") == "no"


# ----- metrics ------------------------------------------------------------- #

def test_exact_match_and_f1():
    assert exact_match("B", "B") == 1
    assert exact_match("B", "C") == 0
    assert exact_match(None, "B") == 0
    assert token_f1("heart failure", "heart failure") == 1.0
    assert 0.0 < token_f1("acute heart failure", "heart failure") < 1.0


def test_score_aggregates_with_coverage():
    preds = ["A", "B", None, "D"]
    golds = ["A", "C", "C", "D"]
    s = score(preds, golds)
    assert s.n == 4
    assert s.em == 0.5            # A,D correct
    assert s.coverage == 0.75     # one miss


# ----- answer-format audit (the anti-artifact gate) ------------------------ #

def test_audit_passes_when_arms_share_schema():
    arms = {
        "No-RAG": ["A", "B", "C", "D"],
        "Hybrid-CARRF": ["A", "A", "C", "D"],
    }
    rep = audit(arms, "mcq")
    assert rep.passed
    assert rep.gate_emit_em_f1()


def test_audit_fails_on_low_coverage():
    arms = {"BM25": ["A", None, None, None]}
    rep = audit(arms, "mcq")
    assert not rep.passed
    assert any("coverage" in r for r in rep.reasons)


def test_audit_fails_on_out_of_schema_labels():
    # An arm emitting free text that normalizes outside {A..D}.
    arms = {"weird": ["A", "B", "yes", "C"]}
    # 'yes' won't normalize as mcq, so simulate a raw schema leak directly:
    rep = audit({"weird": ["A", "B", "E", "C"]}, "mcq")
    assert not rep.passed
    assert any("outside expected" in r for r in rep.reasons)


def test_audit_free_form_has_no_schema_gate():
    rep = audit({"a": ["some long answer", "another"]}, "free")
    assert rep.passed  # free-form is scored by F1/semantic, not a closed schema


# ----- prompts ------------------------------------------------------------- #

def test_build_messages_mcq_includes_format_instruction():
    msgs = prompts.build_messages(
        "What is X?", "mcq", options={"A": "a", "B": "b", "C": "c", "D": "d"}
    )
    assert msgs[0]["role"] == "system"
    assert "single capital letter" in msgs[1]["content"]
    assert "Options:" in msgs[1]["content"]


def test_prompt_set_hash_is_stable():
    assert prompts.prompt_set_hash() == prompts.prompt_set_hash()
    assert prompts.answer_type_for("MCQ") == "mcq"
    assert prompts.answer_type_for("yes/no") == "yes_no"
