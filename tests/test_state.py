import mgr.state as st
from mgr.state import Status


def test_legal_transitions():
    assert st.can_transition(Status.PENDING, Status.READY)
    assert st.can_transition(Status.READY, Status.RUNNING)
    assert st.can_transition(Status.RUNNING, Status.DONE)
    assert st.can_transition(Status.RUNNING, Status.FAILED)
    assert st.can_transition(Status.FAILED, Status.RUNNING)  # resume


def test_illegal_transitions():
    assert not st.can_transition(Status.DONE, Status.RUNNING)  # Done is terminal
    assert not st.can_transition(Status.PENDING, Status.DONE)
    assert not st.can_transition(Status.READY, Status.DONE)


def test_gate_name_normalization():
    assert st.gate_name("H2 (harness)") == "H2"
    assert st.gate_name("G3 (grounded graph)") == "G3"
    assert st.gate_name("P3 gate + oracle labels") == "P3"


def test_resolve_status_gate_satisfied_to_ready():
    ledger = {"H2": True, "G3": False, "P3": False}
    assert st.resolve_status(Status.PENDING, "H2 (harness)", ledger) == Status.READY
    assert st.resolve_status(Status.PENDING, "G3 (grounded graph)", ledger) == Status.BLOCKED


def test_resolve_status_passthrough_for_runner_owned():
    ledger = {"H2": True}
    assert st.resolve_status(Status.DONE, "H2 (harness)", ledger) == Status.DONE
    assert st.resolve_status(Status.RUNNING, "H2 (harness)", ledger) == Status.RUNNING


def test_claim_lifecycle(tmp_path):
    root = tmp_path / "results"
    c = st.acquire_claim("R0001", results_root=root)
    assert c is not None
    # second attempt while live -> None
    assert st.acquire_claim("R0001", results_root=root) is None
    # stale takeover
    c2 = st.acquire_claim("R0001", results_root=root, stale_after_s=0.0)
    assert c2 is not None
    st.release_claim("R0001", results_root=root)
    assert st.read_claim("R0001", results_root=root) is None
