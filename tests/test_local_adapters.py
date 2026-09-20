"""Local judge / extractor, and their parity with the hosted ones."""

import pytest

from mgr.clients.judge_prompts import MAX_TOKENS
from mgr.clients.local_adapters import LocalEntityExtractor, LocalGroundingJudge
from mgr.clients.nim_adapters import NimEntityExtractor, NimGroundingJudge
from mgr.graph.build import build_graph
from mgr.graph.store import InMemoryGraphStore
from mgr.graph.umls import UMLSLinker
from mgr.metrics.grounding_ragas import GroundingItem, score


def canned(prompt: str) -> str:
    """The same prompt-aware replies tests/test_nim_adapters.py uses."""
    p = prompt.lower()
    if "atomic factual claims" in p:
        return "claim one\nclaim two"
    if "fully supported" in p:
        return "yes" if "aspirin" in p else "no"
    if "relevant to answering" in p:
        return "yes" if "myocardial" in p else "no"
    if "how relevant" in p:
        return "0.8"
    if "medical entities" in p:
        return "- aspirin\n- myocardial infarction"
    return ""


class FakeLocal:
    """A local chat client: complete_text -> (text, usage). Records every call."""

    model_id = "fake-local"

    def __init__(self):
        self.calls: list[dict] = []

    def complete_text(self, model, messages, **params):
        prompt = messages[-1]["content"]
        self.calls.append({"model": model, "messages": messages, "params": params})
        return canned(prompt), {"in": 20, "out": 5}


class FakeNim:
    """The hosted shape: judge(model, messages, **params) -> chat envelope."""

    def __init__(self):
        self.calls: list[dict] = []

    def judge(self, model, messages, **params):
        self.calls.append({"model": model, "messages": messages, "params": params})
        return {"choices": [{"message": {"content": canned(messages[-1]["content"])}}]}


# -- the judge behaves like the hosted one -----------------------------------

def test_decompose_entails_relevant_relevance():
    j = LocalGroundingJudge(FakeLocal())
    assert j.decompose("some answer") == ["claim one", "claim two"]
    assert j.entails("aspirin helps", "aspirin reduces clots") is True
    assert j.entails("insulin cures htn", "unrelated text") is False
    assert j.relevant("myocardial infarction tx", "passage about myocardial stuff") is True
    assert j.relevance("q", "a") == 0.8


def test_empty_answer_decomposes_to_nothing_without_a_call():
    client = FakeLocal()
    assert LocalGroundingJudge(client).decompose("   ") == []
    assert client.calls == []


def test_local_and_hosted_judges_agree_on_identical_inputs():
    """Prompt parity: if these drift, a NIM-judged run and a locally-judged run
    are not comparable and the RAGAS columns stop meaning one thing."""
    local, nim = LocalGroundingJudge(FakeLocal()), NimGroundingJudge(FakeNim())
    assert local.decompose("x") == nim.decompose("x")
    assert local.entails("aspirin helps", "p") == nim.entails("aspirin helps", "p")
    assert local.relevant("myocardial", "p") == nim.relevant("myocardial", "p")
    assert local.relevance("q", "a") == nim.relevance("q", "a")


def test_both_judges_send_the_same_prompt_text():
    lc, nc = FakeLocal(), FakeNim()
    LocalGroundingJudge(lc).entails("claim", "premise")
    NimGroundingJudge(nc).entails("claim", "premise")
    assert lc.calls[0]["messages"] == nc.calls[0]["messages"]


# -- the truncation guard ----------------------------------------------------

def test_every_call_sets_its_own_output_budget():
    """The local client defaults to max_new_tokens=8 for MCQ letters. A judge
    that inherits that returns a truncated claim list, which is how the first POC
    reported faithfulness 0.15."""
    client = FakeLocal()
    j = LocalGroundingJudge(client)
    j.decompose("answer")
    j.entails("h", "p")
    j.relevance("q", "a")

    budgets = [c["params"]["max_tokens"] for c in client.calls]
    assert budgets == [MAX_TOKENS["decompose"], MAX_TOKENS["entails"], MAX_TOKENS["relevance"]]
    assert MAX_TOKENS["decompose"] >= 256  # room to finish a claim list


def test_judging_is_deterministic():
    client = FakeLocal()
    LocalGroundingJudge(client).relevance("q", "a")
    assert client.calls[0]["params"]["temperature"] == 0.0


def test_token_spend_is_accounted_separately():
    j = LocalGroundingJudge(FakeLocal())
    j.decompose("a")
    j.entails("h", "p")
    assert j.usage_tokens == {"in": 40, "out": 10}


# -- extractor ---------------------------------------------------------------

def test_entity_extractor_parses_bullets():
    ex = LocalEntityExtractor(FakeLocal())
    assert ex.extract("patient on aspirin after MI") == ["aspirin", "myocardial infarction"]
    assert ex.extract("") == []


def test_extractor_matches_the_hosted_one():
    assert (
        LocalEntityExtractor(FakeLocal()).extract("aspirin")
        == NimEntityExtractor(FakeNim()).extract("aspirin")
    )


def test_extractor_satisfies_the_graph_build_contract():
    store = InMemoryGraphStore()
    linker = UMLSLinker({"aspirin": "C0004057", "myocardial infarction": "C0027051"}, fuzzy_enabled=False)
    report = build_graph(
        [{"id": "c1", "text": "patient on aspirin after MI"}],
        store, LocalEntityExtractor(FakeLocal()), linker,
    )
    assert report.n_concepts == 2
    assert store.concepts_of("c1") == {"C0004057", "C0027051"}


# -- end to end through RAGAS -------------------------------------------------

def test_judge_satisfies_the_grounding_judge_protocol():
    scores = score(
        [GroundingItem(
            question="myocardial infarction tx",
            answer="aspirin reduces mortality",
            contexts=["aspirin after myocardial infarction"],
            reference="aspirin is indicated",
        )],
        LocalGroundingJudge(FakeLocal()),
    )
    assert scores.n == 1
    assert scores.answer_relevance == pytest.approx(0.8)
    assert 0.0 <= scores.faithfulness <= 1.0
    assert 0.0 <= scores.context_precision <= 1.0
