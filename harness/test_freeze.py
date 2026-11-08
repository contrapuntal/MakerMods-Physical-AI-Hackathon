"""Freeze control, state.json and the early SIGUSR1 handler (stdlib only; no robot needed)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.freeze import FreezeControl, FreezeRequested, StateFile, first_alarm, hold_measured

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def ctl():
    old = signal.getsignal(signal.SIGUSR1)
    c = FreezeControl()
    c.install_freeze_signal()
    yield c
    signal.signal(signal.SIGUSR1, old)


def test_sigusr1_sets_freeze_and_clears_end(ctl):
    calls = []
    ctl.on_freeze = lambda: calls.append(1)
    ctl.end_requested.set()
    os.kill(os.getpid(), signal.SIGUSR1)
    assert ctl.frozen
    assert not ctl.end_requested.is_set()
    assert calls == [1]


def test_interruptible_sleep_raises_after_freeze(ctl):
    threading.Timer(0.05, ctl.freeze.set).start()
    t0 = time.monotonic()
    with pytest.raises(FreezeRequested):
        ctl.interruptible_sleep(5)
    assert time.monotonic() - t0 < 0.5


def test_begin_end_allows_refreeze(ctl):
    os.kill(os.getpid(), signal.SIGUSR1)
    ctl.begin_end()
    ctl.interruptible_sleep(0.01)
    os.kill(os.getpid(), signal.SIGUSR1)
    with pytest.raises(FreezeRequested):
        ctl.interruptible_sleep(0.01)


def test_state_file_atomic_and_schema(tmp_path):
    sf = StateFile(tmp_path, link=tmp_path / "link")
    sf.update(state="starting", pid=os.getpid(), started_at=1.0, dry_run=True, freeze_supported=True,
              stage=None, score=None, alarm=None, error=None, version=1)
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["version"] == 1
    for key in ("state", "pid", "started_at", "stage", "dry_run", "freeze_supported", "score", "alarm",
                "error", "updated_at"):
        assert key in data
    assert not list(tmp_path.glob("*.tmp"))


def test_link_current_points_at_run_dir(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    link = tmp_path / "current"
    sf = StateFile(out, link=link)
    sf.link_current()
    sf.link_current()
    assert link.resolve() == out.resolve()


def test_hold_measured_writes_measured_once():
    written = []
    arm = SimpleNamespace(read=lambda: SimpleNamespace(positions_rad=[1, 2], gripper=0.3), write=written.append)
    hold_measured({"left": arm}, lambda pos, grip: ("cmd", list(pos), grip))
    assert written == [("cmd", [1, 2], 0.3)]


def test_first_alarm():
    ok = SimpleNamespace(alarm=lambda: None, is_alive=lambda: True)
    hot = SimpleNamespace(alarm=lambda: "over temp", is_alive=lambda: True)
    dead = SimpleNamespace(alarm=lambda: None, is_alive=lambda: False)
    assert first_alarm({"left": hot, "right": ok}) == "left: over temp"
    assert first_alarm({"left": ok, "right": dead}) == "right: not answering"
    assert first_alarm({"left": ok}) is None


def test_main_installs_freeze_before_heavy_imports():
    code = (
        "import sys; sys.modules['cv2'] = None\n"
        "import harness.__main__ as m\n"
        "try:\n"
        "    m.main(['run', '--help'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0\n"
        "assert sys.modules.get('cv2') is None\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert "SIGUSR1 freezes the arms" in " ".join(res.stdout.split())
