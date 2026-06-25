from mgr import ids


def test_run_id_roundtrip():
    for i in (1, 42, 123, 244):
        assert ids.run_id_index(ids.format_run_id(i)) == i
    assert ids.format_run_id(123) == "R0123"


def test_run_id_rejects_noncanonical():
    for bad in ("R1", "123", "RXYZ", "r0123", "R00123"):
        try:
            ids.run_id_index(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_slug_roundtrip():
    s = ids.slug("Hybrid-CARRF-CARe", "MedQA-US", "Llama-70B", 42)
    assert s == "Hybrid-CARRF-CARe__MedQA-US__Llama-70B__s42"
    parts = ids.parse_slug(s)
    assert parts == {
        "condition": "Hybrid-CARRF-CARe",
        "benchmark": "MedQA-US",
        "backbone": "Llama-70B",
        "seed": "42",
    }


def test_config_hash_is_order_insensitive_for_dict_keys():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert ids.config_hash(a) == ids.config_hash(b)


def test_config_hash_is_order_sensitive_for_lists():
    a = {"retrievers": ["bm25", "dense"]}
    b = {"retrievers": ["dense", "bm25"]}
    assert ids.config_hash(a) != ids.config_hash(b)


def test_result_paths():
    assert ids.record_path("R0123").as_posix().endswith("results/per-run/R0123.json")
    assert ids.items_path("R0123").as_posix().endswith("results/per-run/R0123/items.jsonl")
    assert ids.claim_path("R0123").as_posix().endswith("results/per-run/R0123/.claim")
