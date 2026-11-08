"""run_match with Dream's `robots`/`dream` packages stubbed (never imports the real ones, opens no hardware).

`stub_dream(monkeypatch)` returns a StubWorld that later tests reuse. It records the call order in `world.calls`
(open_arm, home, torque_off, write, lock_acquired, lock_released, view_opened, view_closed), lets a test choose how
the control loop ends (`world.loop_error`, or `world.loop_gate` to hold it until the test sets it) and replaces the
vision, the relay and the supervisor with trivial stand-ins that finish the relay at once (`world.finish = False`
keeps the supervisor waiting for a stop).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.config import MatchConfig
from harness.freeze import FreezeControl, StateFile

# harness.claude_check imports anthropic at module level and the main env lacks it: import dream_executor with a
# stand-in, removed again afterwards so it does not leak into other tests.
_saved = sys.modules.get("anthropic")
if _saved is None:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        sys.modules["anthropic"] = types.ModuleType("anthropic")
from harness import dream_executor  # noqa: E402
if _saved is None and isinstance(sys.modules.get("anthropic"), types.ModuleType) \
        and not hasattr(sys.modules["anthropic"], "Anthropic"):
    del sys.modules["anthropic"]


class StubWorld:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.states: list[dict] = []  # every state.json write, in order
        self.lock_error: Exception | None = None
        self.view_error: OSError | None = None
        self.lock_args: list = []
        self.view_ports: list[int] = []
        self.watched: list = []
        self.view_clients: list = []
        self.loop_error: Exception | None = None
        self.loop_gate: asyncio.Event | None = None  # ControlLoop.run() waits on it when set
        self.finish = True  # the stub supervisor ends the relay at once
        self.arms: dict = {}
        self.home_sleeps = 0
        self.client = SimpleNamespace(name="client", wait_for_ik=lambda t: True)


class StubArm:
    home_rad = [0.0]

    def __init__(self, world: StubWorld, name: str) -> None:
        self.world, self.name = world, name

    def write(self, cmd) -> None:
        self.world.calls.append("write")

    def torque_off(self) -> None:
        self.world.calls.append("torque_off")


class StubRelay:
    stage = "unplug"
    step = SimpleNamespace(instruction="unplug cables", kind="unplug")
    passed = {"unplug": 3.0}
    own_in_bin = other_in_bin = 0
    placed = None
    events: list = []
    cal = None

    def __init__(self, match) -> None:
        pass

    def start(self, world, t) -> None:
        pass

    def next_step(self, world, t):
        return self.step

    def score(self):
        return 10

    def time_left(self, now):
        return 100.0

    def log(self, t, what) -> None:
        pass


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def stub_dream(monkeypatch) -> StubWorld:
    world = StubWorld()

    class ArmError(Exception): ...
    class SafetyStop(Exception): ...
    class CameraError(Exception): ...

    errors = _module("robots.errors", ArmError=ArmError, SafetyStop=SafetyStop, CameraError=CameraError,
                     StationError=type("StationError", (Exception,), {}))

    @contextlib.contextmanager
    def holding(path, holder):
        world.lock_args.append((path, holder))
        if world.lock_error is not None:
            raise world.lock_error
        world.calls.append("lock_acquired")
        try:
            yield
        finally:
            world.calls.append("lock_released")

    arms_lock = _module("robots.arms_lock", holding=holding, LOCK_NAME="arms.lock")

    class ControlLoop:
        task = ""
        video: dict = {}
        state = None

        def __init__(self, **kw) -> None:
            self.task = kw["task"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self) -> None:
            if world.loop_error is not None:
                raise world.loop_error
            if world.loop_gate is not None:
                await world.loop_gate.wait()
            else:
                await asyncio.Event().wait()  # until cancelled

        def set_task(self, task) -> None:
            self.task = task

    control = _module("robots.control", ControlLoop=ControlLoop,
                      run_teardown=lambda what, fn: fn())

    class FollowerArm:
        def __init__(self, name, arm, home_rad=None) -> None:
            self.arm = arm

        def torque_off(self) -> None:
            self.arm.torque_off()

    followers = _module("robots.followers", FollowerArm=FollowerArm)

    def return_to_home(arms, *, duration_s=0.0, sleep=None, **kw) -> bool:
        world.calls.append("home")
        for _ in range(3):
            (sleep or (lambda dt: None))(0.0)
            world.home_sleeps += 1
        return True

    home = _module("robots.home", return_to_home=return_to_home, HOME_DURATION_S=1.0)

    def open_client(cfg):
        world.calls.append("open_client")
        return world.client

    rrun = _module(
        "robots.run",
        make_cameras=lambda app: [SimpleNamespace(name="scene", frame=SimpleNamespace(data=None, timestamp_s=0.0),
                                                  start=lambda: None, stop=lambda: None,
                                                  set_exposure_hold=lambda on: None)],
        make_kinematics=lambda app: None, open_client=open_client, DryRun=lambda arm: arm,
        loop_settings=lambda app: {}, policy_config=lambda app, checkpoint: {"checkpoint": checkpoint},
        with_task=lambda cfg, task: cfg, TRAJ_WARMUP_S=1.0)

    def open_arm(cfg, name):
        world.calls.append("open_arm")
        arm = StubArm(world, name)
        world.arms[name] = arm
        return arm

    bodies = _module("robots.bodies", for_config=lambda app: SimpleNamespace(
        open_arm=open_arm, open_arm_read_only=None))
    rconfig = _module("robots.config", load=lambda path: SimpleNamespace(
        follower_l=None, follower_r=None, inference_log_path=None))
    robots = _module("robots", config=rconfig, control=control, followers=followers, home=home, run=rrun,
                     bodies=bodies, errors=errors, arms_lock=arms_lock)
    robots.__path__ = []

    class DebugView:
        def __init__(self, client, port=0) -> None:
            if world.view_error is not None:
                raise world.view_error
            world.calls.append("view_opened")
            world.view_ports.append(port)
            world.view_clients.append(client)
            self.url = f"http://127.0.0.1:{port}/"

        def watch(self, client) -> None:
            world.watched.append(client)

        def close(self) -> None:
            world.calls.append("view_closed")

    dream = _module("dream", DebugView=DebugView)

    for mod in (robots, rconfig, control, followers, home, rrun, bodies, errors, arms_lock, dream):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)

    # Stand-ins for the vision, the relay, the supervisor and the report: this file tests run_match's wiring.
    fake_vision = SimpleNamespace(locate_layout=lambda img, cal: SimpleNamespace(note=lambda: "stub"),
                                  with_layout=lambda cal, layout: SimpleNamespace(bin_roi=None, hub_roi=None),
                                  detect=lambda *a: None, draw=lambda *a: None)
    monkeypatch.setattr(dream_executor, "vision", fake_vision)
    monkeypatch.setattr(dream_executor.cv2, "imwrite", lambda *a: True)
    monkeypatch.setattr(dream_executor, "Relay", StubRelay)
    monkeypatch.setattr(dream_executor, "ClaudeVerifier",
                        lambda: SimpleNamespace(probe=lambda: None, available=False))
    monkeypatch.setattr(dream_executor, "_write_report", lambda relay, out_dir: None)

    async def supervise(loop, relay, match, stop, out_dir, claude, go_rgb=None, **kw):
        if world.finish:
            await asyncio.sleep(0)
            stop.set()
        await stop.wait()

    monkeypatch.setattr(dream_executor, "_supervise", supervise)
    return world


class RecordingState(StateFile):
    def __init__(self, out_dir: Path, world: StubWorld) -> None:
        super().__init__(out_dir, link=out_dir / "link")
        self.world = world

    def update(self, **fields):
        out = super().update(**fields)
        self.world.states.append(out)
        return out


def run(world: StubWorld, tmp_path: Path, **kw):
    state = RecordingState(tmp_path, world)
    match = MatchConfig(own_color="white")
    coro = dream_executor.run_match(match, "ckpt", dry_run=False, out_dir=tmp_path, ctl=FreezeControl(),
                                    state=state, **kw)
    return asyncio.run(coro)


def test_holds_arms_lock_for_whole_run(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    calls = stub_dream.calls
    assert calls[0] == "lock_acquired"
    assert calls[-1] == "lock_released"
    assert calls.index("lock_released") > max(i for i, c in enumerate(calls) if c == "torque_off")
    assert calls.index("open_arm") > calls.index("lock_acquired")
    path, holder = stub_dream.lock_args[0]
    assert path == Path("~/.dreamlabs/arms.lock").expanduser() and holder == "harness run"


def test_lock_held_elsewhere_fails_cleanly(stub_dream, tmp_path):
    stub_dream.lock_error = RuntimeError("the arms are held by 1234 dreamcli robot run")
    with pytest.raises(RuntimeError, match="dreamcli robot run"):
        run(stub_dream, tmp_path)
    assert "open_arm" not in stub_dream.calls
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "dreamcli robot run" in final["error"]


def test_debugview_opened_on_8093_and_closed(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    assert stub_dream.view_ports == [8093]
    assert stub_dream.view_clients == [stub_dream.client]
    calls = stub_dream.calls
    assert calls.index("open_client") < calls.index("view_opened") < calls.index("view_closed")


def test_debugview_oserror_is_a_warning(stub_dream, tmp_path, caplog):
    stub_dream.view_error = OSError("port in use")
    with caplog.at_level(logging.WARNING):
        run(stub_dream, tmp_path)
    assert "view_closed" not in stub_dream.calls and "open_arm" in stub_dream.calls
    assert any("8093" in r.message for r in caplog.records)


def test_state_transitions_normal_end(stub_dream, tmp_path):
    run(stub_dream, tmp_path)
    names = [s["state"] for s in stub_dream.states]
    dedup = [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]
    assert dedup == ["running", "homing", "parking", "done"]
    assert stub_dream.states[-1]["score"] == 10
    assert type(stub_dream.states[-1]["score"]) is int


def test_exception_writes_error_state(stub_dream, tmp_path):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("loop blew up")
    with pytest.raises(RuntimeError, match="loop blew up"):
        run(stub_dream, tmp_path)
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["state"] == "error" and "loop blew up" in final["error"]
    calls = stub_dream.calls
    assert calls.count("torque_off") == 2
    assert calls.index("lock_released") > max(i for i, c in enumerate(calls) if c == "torque_off")


def test_report_failure_still_releases_lock_and_writes_error(stub_dream, tmp_path, monkeypatch):
    def boom(relay, out_dir):
        raise OSError("disk full")
    monkeypatch.setattr(dream_executor, "_write_report", boom)
    with pytest.raises(OSError):
        run(stub_dream, tmp_path)
    assert stub_dream.calls[-1] == "lock_released"
    assert stub_dream.states[-1]["state"] == "error"


def test_view_follows_restarted_client(stub_dream, tmp_path):
    stub_dream.finish = False
    stub_dream.loop_error = RuntimeError("stalled")
    with pytest.raises(RuntimeError):
        run(stub_dream, tmp_path)
    assert stub_dream.watched == [stub_dream.client] * 3  # one per restart


def test_real_supervise_writes_stage(tmp_path, monkeypatch):
    # The real _supervise (the fixture replaces it): a stage change must reach state.json, although the loop's
    # own `state` is a robot state.
    monkeypatch.setattr(dream_executor, "POLL_S", 0.001)
    monkeypatch.setattr(dream_executor.vision, "detect", lambda *a: None)
    frame = SimpleNamespace(data=None, timestamp_s=0.0)
    arm = SimpleNamespace(gripper=0.0)
    loop = SimpleNamespace(video={"scene": frame}, state=SimpleNamespace(left=arm, right=arm), task="t")
    relay = SimpleNamespace(stage="unplug", confirm_kinds=set(), pending_confirm=None, step=None,
                            observe=lambda *a: dream_executor.Outcome.RUNNING)
    statefile = StateFile(tmp_path, link=tmp_path / "link")
    stop = asyncio.Event()

    async def go():
        task = asyncio.create_task(dream_executor._supervise(
            loop, relay, SimpleNamespace(calibration=None), stop, tmp_path, SimpleNamespace(available=False), statefile=statefile))
        await asyncio.sleep(0.05)
        relay.stage = "stack"
        await asyncio.sleep(0.05)
        stop.set()
        await task

    asyncio.run(go())
    assert json.loads((tmp_path / "state.json").read_text())["stage"] == "stack"
