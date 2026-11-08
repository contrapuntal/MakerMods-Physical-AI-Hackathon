"""The executor's model switch, on a stand-in for Dream's `robots` package (no arms, cameras or network)."""

import asyncio
import sys
import types

import numpy as np

from harness import relay as relay_mod
from harness.config import DEFAULT_CALIBRATION, Calibration, MatchConfig

LOOPS: list = []  # (model, first task) per ControlLoop the executor opened


class _Frame:
    def __init__(self):
        self.data = np.full((360, 640, 3), 90, np.uint8)
        self.timestamp_s = 0.0


class _Reader:
    def __init__(self, name):
        self.name, self.frame = name, _Frame()

    def start(self): pass
    def stop(self): pass
    def set_exposure_hold(self, hold): pass


class _Client:
    def __init__(self, model):
        self.model = model

    def wait_for_ik(self, s):
        return True


class _Loop:
    def __init__(self, *, client, task, **kw):
        self.client, self.task = client, task
        LOOPS.append((client.model, task))
        self.video = types.SimpleNamespace(get=lambda name: _Frame())
        g = types.SimpleNamespace(gripper=1.0)
        self.state = types.SimpleNamespace(left=g, right=g)

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def run(self):
        while True:
            await asyncio.sleep(0.05)

    def set_task(self, task):
        self.task = task


def _fake_robots(monkeypatch):
    errors = types.SimpleNamespace(ArmError=type("ArmError", (Exception,), {}),
                                   SafetyStop=type("SafetyStop", (Exception,), {}),
                                   CameraError=type("CameraError", (Exception,), {}))
    app = types.SimpleNamespace(follower_l=None, follower_r=None, inference_log_path=None)
    arm = types.SimpleNamespace(home_rad=None, torque_off=lambda: None)
    mods = {
        "robots": types.ModuleType("robots"),
        "robots.config": types.SimpleNamespace(load=lambda p: app),
        "robots.control": types.SimpleNamespace(ControlLoop=_Loop, run_teardown=lambda name, fn: fn()),
        "robots.followers": types.SimpleNamespace(FollowerArm=lambda *a, **k: arm),
        "robots.home": types.SimpleNamespace(return_to_home=lambda arms, duration_s: True, HOME_DURATION_S=0.0),
        "robots.run": types.SimpleNamespace(
            with_task=lambda a, t: a, policy_config=lambda a, model: model, open_client=_Client, TRAJ_WARMUP_S=1.0,
            make_cameras=lambda a: [_Reader("scene"), _Reader("wrist_l"), _Reader("wrist_r")],
            make_kinematics=lambda a: None, loop_settings=lambda a: {}, DryRun=lambda a: a),
        "robots.bodies": types.SimpleNamespace(
            for_config=lambda a: types.SimpleNamespace(open_arm=lambda cfg, name: arm, open_arm_read_only=None)),
        "robots.errors": errors,
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for name in ("config", "control", "followers", "home", "run", "bodies", "errors"):
        setattr(mods["robots"], name, mods[f"robots.{name}"])


def test_each_stage_runs_on_its_own_model(monkeypatch, tmp_path):
    _fake_robots(monkeypatch)
    monkeypatch.setattr("harness.claude_check._load_key", lambda: None)  # Claude off
    LOOPS.clear()

    def observe(self, world, t, grip):  # every unplug / stack step passes at once; balls run on to the clock
        if self.step.stage in ("unplug", "stack"):
            self.step.kind = "unplug" if self.step.stage == "unplug" else "stack_all"
            self._last_grip = dict(grip)
            self._on_pass(self.step, world, t)
            return relay_mod.Outcome.PASSED
        return relay_mod.Outcome.FAILED if self.time_left(t) < self.m.stop_margin_s else relay_mod.Outcome.RUNNING

    monkeypatch.setattr(relay_mod.Relay, "observe", observe)
    from harness.dream_executor import run_match
    m = MatchConfig(own_color="orange", calibration=Calibration.load(DEFAULT_CALIBRATION), time_limit_s=4.0, stop_margin_s=1.0,
                    stage_models={"unplug": "ft-cables", "stack": "ft-stack", "balls": "ft-balls"},
                    stage_texts={"unplug": "U", "stack": "S", "balls": "B"})
    r = asyncio.run(run_match(m, "base", dry_run=True, out_dir=tmp_path))
    assert [model for model, _ in LOOPS] == ["ft-cables", "ft-stack", "ft-balls"]
    # Each new model starts on its own stage's words, every time (no park sentence even with no ball seen).
    assert [task for _, task in LOOPS] == ["U", "S", "B"]
    assert list(r.passed) == ["unplug", "stack"]


def test_one_model_never_switches(monkeypatch, tmp_path):
    _fake_robots(monkeypatch)
    monkeypatch.setattr("harness.claude_check._load_key", lambda: None)
    LOOPS.clear()

    def observe(self, world, t, grip):
        if self.step.stage == "unplug":
            self._last_grip = dict(grip)
            self._on_pass(self.step, world, t)
            return relay_mod.Outcome.PASSED
        return relay_mod.Outcome.FAILED if self.time_left(t) < self.m.stop_margin_s else relay_mod.Outcome.RUNNING

    monkeypatch.setattr(relay_mod.Relay, "observe", observe)
    from harness.dream_executor import run_match
    m = MatchConfig(own_color="orange", calibration=Calibration.load(DEFAULT_CALIBRATION), time_limit_s=3.0, stop_margin_s=1.0,
                    stage_texts={"unplug": "U", "stack": "S"})
    asyncio.run(run_match(m, "ft-all", dry_run=True, out_dir=tmp_path))
    assert LOOPS == [("ft-all", "U")]  # one loop the whole match: the stack step only changed its task


def test_stacked_tower_is_released_then_checked_standing(monkeypatch, tmp_path):
    _fake_robots(monkeypatch)
    LOOPS.clear()
    asked = []

    class FakeClaude:
        available = True

        def probe(self):
            return None

        def ask(self, question, rgb, before=None, extra=None):
            asked.append(question)
            if question == "unplugged_since":
                return True, "cable out"
            return (question in ("stacked_now", "standing_now")), "stub"

    import harness.dream_executor as ex
    monkeypatch.setattr(ex, "ClaudeVerifier", FakeClaude)
    monkeypatch.setattr(relay_mod, "STACK_ALL_MIN_S", 0.0)
    m = MatchConfig(own_color="orange", calibration=Calibration.load(DEFAULT_CALIBRATION), time_limit_s=12.0,
                    stop_margin_s=1.0, stack_pass="3-cubes",
                    stage_models={"unplug": "ft-cables", "stack": "ft-stack", "balls": "ft-balls"},
                    stage_texts={"unplug": "U", "stack": "S", "balls": "B"})
    r = asyncio.run(ex.run_match(m, "base", dry_run=True, out_dir=tmp_path))
    assert "stacked_now" in asked and "standing_now" in asked
    assert list(r.passed)[:2] == ["unplug", "stack"]
    assert [model for model, _ in LOOPS][:3] == ["ft-cables", "ft-stack", "ft-balls"]
