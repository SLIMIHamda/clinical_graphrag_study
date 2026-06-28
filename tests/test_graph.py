from mgr.graph.build import build_graph, query_concept_fn
from mgr.graph.store import InMemoryGraphStore
from mgr.graph.umls import UMLSLinker
from mgr.retrieval.graph_retriever import GraphRetriever

EXACT = {"aspirin": "C1", "myocardial infarction": "C2", "diabetes": "C3", "metformin": "C4"}


class FakeExtractor:
    """Returns whichever known surface forms appear in the text."""

    def extract(self, text):
        t = text.lower()
        return [k for k in EXACT if k in t]


def _chunks():
    return [
        {"id": "ch1", "text": "aspirin for myocardial infarction", "source": "sp", "layer": "chunk"},
        {"id": "ch2", "text": "metformin for diabetes", "source": "sp", "layer": "chunk"},
        {"id": "ch3", "text": "myocardial infarction management", "source": "tb", "layer": "chunk"},
    ]


def _built_store():
    store = InMemoryGraphStore()
    linker = UMLSLinker(EXACT)
    report = build_graph(_chunks(), store, FakeExtractor(), linker)
    return store, report


def test_build_populates_graph_with_provenance():
    store, report = _built_store()
    assert report.n_chunks == 3
    assert report.n_concepts == 4
    assert report.graph_hash  # non-empty hash
    assert store.chunks["ch1"]["source"] == "sp"   # provenance kept (no blank scope)
    assert store.concepts_of("ch1") == {"C1", "C2"}


def test_build_hash_is_deterministic():
    s1, r1 = _built_store()
    s2, r2 = _built_store()
    assert r1.graph_hash == r2.graph_hash


def test_chunks_for_concepts_ranks_by_overlap():
    store, _ = _built_store()
    # query concepts {C2 (MI)} -> ch1 and ch3 mention it, ch2 does not
    hits = dict(store.chunks_for_concepts({"C2"}, hops=0, limit=10))
    assert set(hits) == {"ch1", "ch3"}
    assert "ch2" not in hits


def test_neighbour_expansion_adds_related_chunks():
    store, _ = _built_store()
    store.link_concept_concept("C2", "C3")  # MI related to diabetes
    hits = dict(store.chunks_for_concepts({"C2"}, hops=1, limit=10))
    assert "ch2" in hits                      # reached via C2->C3->metformin/diabetes chunk
    assert hits["ch1"] > hits["ch2"]          # direct concept weighs more than neighbour


def test_graph_retriever_end_to_end():
    store, _ = _built_store()
    linker = UMLSLinker(EXACT)
    qfn = query_concept_fn(FakeExtractor(), linker)
    gr = GraphRetriever(store, qfn, hops=0)
    res = gr.retrieve("treatment of myocardial infarction", depth_k=2)
    assert res.retrieved_ids[0] in {"ch1", "ch3"}
    assert res.context is not None
    assert res.ranks[res.retrieved_ids[0]] == 1


def test_graph_retriever_no_concepts_returns_empty():
    store, _ = _built_store()
    gr = GraphRetriever(store, lambda q: set())
    res = gr.retrieve("unrelated", depth_k=5)
    assert res.retrieved_ids == []
    assert res.context is None
