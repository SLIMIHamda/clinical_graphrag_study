"""Offline graph path: dictionary extraction + snapshot persistence.

Together these make the graph arms reachable without a hosted LLM (for query-time
concept grounding) and without Neo4j (for keeping the graph past process exit).
"""

import json

import pytest

from mgr.graph.build import build_graph, query_concept_fn
from mgr.graph.extract import NgramExtractor
from mgr.graph.store import InMemoryGraphStore, load_snapshot, save_snapshot
from mgr.graph.umls import UMLSLinker

VOCAB = {
    "heart failure": "C0018801",
    "heart": "C0018787",
    "metformin": "C0025598",
    "type 2 diabetes": "C0011860",
}


# -- extraction -------------------------------------------------------------

def test_longest_match_wins():
    ex = NgramExtractor.from_linker_dict(VOCAB)
    assert ex.extract("history of heart failure") == ["heart failure"]


def test_shorter_term_still_matches_on_its_own():
    ex = NgramExtractor.from_linker_dict(VOCAB)
    assert ex.extract("the heart is enlarged") == ["heart"]


def test_multiple_mentions_in_order():
    ex = NgramExtractor.from_linker_dict(VOCAB)
    got = ex.extract("metformin for type 2 diabetes and heart failure")
    assert got == ["metformin", "type 2 diabetes", "heart failure"]


def test_out_of_vocabulary_text_yields_nothing():
    ex = NgramExtractor.from_linker_dict(VOCAB)
    assert ex.extract("the patient was discharged home") == []


def test_max_n_grows_to_fit_the_longest_vocabulary_term():
    ex = NgramExtractor(vocab=["a b c d e f"], max_n=2)
    assert ex.max_n >= 6
    assert ex.extract("x a b c d e f y") == ["a b c d e f"]


def test_unbounded_mode_emits_ngrams_and_drops_bare_stopwords():
    ex = NgramExtractor(max_n=2)
    got = ex.extract("the heart failure")
    assert "the" not in got
    assert "heart failure" in got and "heart" in got


def test_empty_text_is_safe():
    assert NgramExtractor.from_linker_dict(VOCAB).extract("") == []


# -- composition with the linker --------------------------------------------

def test_query_concept_fn_grounds_to_cuis_without_a_model():
    ex = NgramExtractor.from_linker_dict(VOCAB)
    fn = query_concept_fn(ex, UMLSLinker(VOCAB, fuzzy_enabled=False))
    assert fn("worsening heart failure on metformin") == {"C0018801", "C0025598"}


def test_concept_layer_is_actually_populated_over_a_corpus():
    """The failure this replaces: 1 concept / 1 link over 60 chunks, which made
    CA-RRF identical to plain RRF while every structural check still passed."""
    chunks = [
        {"id": "c1", "text": "Patient with heart failure started on metformin."},
        {"id": "c2", "text": "Type 2 diabetes managed with metformin."},
        {"id": "c3", "text": "No relevant findings."},
    ]
    store = InMemoryGraphStore()
    report = build_graph(
        chunks, store, NgramExtractor.from_linker_dict(VOCAB), UMLSLinker(VOCAB, fuzzy_enabled=False)
    )
    assert report.n_chunks == 3
    assert report.n_concepts >= 3
    assert store.concepts_of("c1") == {"C0018801", "C0025598"}
    assert store.concepts_of("c3") == set()


# -- persistence ------------------------------------------------------------

def _built_store():
    chunks = [
        {"id": "c1", "text": "heart failure and metformin", "source": "notes", "layer": "chunk"},
        {"id": "c2", "text": "type 2 diabetes", "source": "notes", "layer": "chunk"},
    ]
    store = InMemoryGraphStore()
    build_graph(
        chunks, store, NgramExtractor.from_linker_dict(VOCAB),
        UMLSLinker(VOCAB, fuzzy_enabled=False),
        concept_relations=[("C0018801", "C0011860")],
    )
    return store


def test_snapshot_round_trip_preserves_graph_hash(tmp_path):
    store = _built_store()
    path = tmp_path / "graph.json"
    written_hash = save_snapshot(store, path)

    reloaded = load_snapshot(path)
    assert reloaded.signature() == written_hash == store.signature()


def test_snapshot_round_trip_preserves_retrieval_behaviour(tmp_path):
    store = _built_store()
    path = tmp_path / "graph.json"
    save_snapshot(store, path)
    reloaded = load_snapshot(path)

    cuis = {"C0018801"}
    assert reloaded.chunks_for_concepts(cuis, hops=1) == store.chunks_for_concepts(cuis, hops=1)
    assert reloaded.text("c1") == store.text("c1")
    assert reloaded.concepts_of("c1") == store.concepts_of("c1")


def test_snapshot_preserves_chunk_provenance(tmp_path):
    store = _built_store()
    path = tmp_path / "graph.json"
    save_snapshot(store, path)
    assert load_snapshot(path).chunks["c1"]["source"] == "notes"


def test_tampered_snapshot_is_rejected(tmp_path):
    """graph_hash is the G3 evidence; a snapshot that no longer matches it must
    not load silently."""
    store = _built_store()
    path = tmp_path / "graph.json"
    save_snapshot(store, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chunk_concepts"]["c1"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(path)
