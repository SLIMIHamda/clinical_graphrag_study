import importlib.util
from pathlib import Path

# pod_watch lives under infra/guards (not a package); load it by path.
_SPEC = importlib.util.spec_from_file_location(
    "pod_watch", Path(__file__).resolve().parent.parent / "infra" / "guards" / "pod_watch.py"
)
pod_watch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pod_watch)


def test_is_idle_threshold():
    assert pod_watch.is_idle(last_heartbeat_ts=0.0, now=2000.0, idle_threshold_s=1800.0)
    assert not pod_watch.is_idle(last_heartbeat_ts=1000.0, now=2000.0, idle_threshold_s=1800.0)


def test_read_heartbeat_missing(tmp_path):
    assert pod_watch.read_heartbeat(tmp_path / "nope") is None


def test_check_once_alive_when_fresh(tmp_path):
    hb = tmp_path / ".heartbeat"
    hb.write_text("x")
    # now == file mtime -> not idle
    action = pod_watch.check_once(hb, tmp_path / "down.sh", idle_threshold_s=1800.0, now=hb.stat().st_mtime)
    assert action == "alive"


def test_check_once_no_heartbeat(tmp_path):
    action = pod_watch.check_once(tmp_path / "missing", tmp_path / "down.sh", idle_threshold_s=10.0)
    assert action == "no-heartbeat-yet"


def test_check_once_tears_down_when_idle(tmp_path, monkeypatch):
    hb = tmp_path / ".heartbeat"
    hb.write_text("x")
    called = {}
    monkeypatch.setattr(pod_watch, "teardown", lambda script: called.setdefault("script", script) or 0)
    action = pod_watch.check_once(
        hb, tmp_path / "down.sh", idle_threshold_s=1.0, now=hb.stat().st_mtime + 9999
    )
    assert action == "torn-down"
    assert "script" in called
