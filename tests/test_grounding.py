import re

import pytest

from mgr.metrics.grounding_ragas import (
    GroundingItem,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    score,
)


def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class FakeJudge:
    """Deterministic stand-in for the NIM judge.

    - decompose: split on ';'
    - entails: every content token of the claim appears in the premise
    - relevant: question and passage share a token
    - relevance: token Jaccard(question, answer)
    """

    def decompose(self, text):
        return [c.strip() for c in text.split(";") if c.strip()]

    def entails(self, hypothesis, premise):
        h, p = _toks(hypothesis), _toks(premise)
        return bool(h) and h <= p

    def relevant(self, question, passage):
        return bool(_toks(question) & _toks(passage))

    def relevance(self, question, answer):
        q, a = _toks(question), _toks(answer)
        if not (q or a):
            return 0.0
        return len(q & a) / len(q | a)


JUDGE = FakeJudge()


def test_faithfulness_full_when_claims_supported():
    ctx = ["aspirin reduces platelet aggregation in myocardial infarction"]
    answer = "aspirin reduces platelet aggregation; myocardial infarction"
    assert faithfulness(answer, ctx, JUDGE) == 1.0


def test_faithfulness_drops_on_hallucinated_claim():
    ctx = ["aspirin reduces platelet aggregation"]
    answer = "aspirin reduces platelet aggregation; insulin cures hypertension"
    assert faithfulness(answer, ctx, JUDGE) == pytest.approx(0.5)


def test_faithfulness_empty_answer_is_vacuously_true():
    assert faithfulness("", ["anything"], JUDGE) == 1.0


def test_answer_relevance_clipped_and_monotone():
    high = answer_relevance("treat myocardial infarction", "treat myocardial infarction", JUDGE)
    low = answer_relevance("treat myocardial infarction", "the capital of france", JUDGE)
    assert high > low
    assert 0.0 <= low <= high <= 1.0


def test_context_precision_rewards_relevant_high_ranks():
    q = "aspirin myocardial infarction"
    good = ["aspirin myocardial infarction trial", "unrelated france text"]
    bad = ["unrelated france text", "aspirin myocardial infarction trial"]
    assert context_precision(q, good, JUDGE) > context_precision(q, bad, JUDGE)


def test_context_recall_measures_gold_coverage():
    gold = "aspirin helps; reperfusion helps"
    ctx = ["aspirin helps in this case"]   # only the first gold claim is covered
    assert context_recall(gold, ctx, JUDGE) == pytest.approx(0.5)


def test_score_aggregates_all_four():
    items = [
        GroundingItem(
            question="aspirin for myocardial infarction",
            answer="aspirin reduces platelet aggregation",
            contexts=["aspirin reduces platelet aggregation in myocardial infarction"],
            reference="aspirin reduces platelet aggregation",
        )
    ]
    s = score(items, JUDGE)
    assert s.n == 1
    assert s.faithfulness == 1.0
    assert s.context_recall == 1.0
    assert 0.0 <= s.answer_relevance <= 1.0
