"""Offline tests of the station server and the DimOS station backend: a fake station stands in for Dream-1.

The fake reacts to each instruction the way a good Dream-1 rollout would look from the overhead camera (the gripper
closes, then opens, and the object moves), so the step logic, the HTTP API, DimOS's DreamStation client and the
referee are all exercised without a robot.

    repos/dimos/.venv/bin/python -m pytest harness/test_station_server.py -q
    # or serve the fake, to drive it from DimOS (dimos_hillclimb serve --backend station):
    repos/dimos/.venv/bin/python -m harness.test_station_server --port 8765
"""

from __future__ import annotations

import re
import threading

import numpy as np
import pytest

from harness.config import Calibration, ColorRange, MatchConfig
from harness.relay import PARK
from harness.station import Snapshot
from harness.station_server import StationServer, make_http
from harness.vision import Obj, World

TOWER = (250.0, 250.0)
BIN = (550.0, 50.0)


def cal() -> Calibration:
    c = Calibration(
        hub_roi=[(0, 0), (100, 0), (100, 100), (0, 100)],
        bin_roi=[(500, 0), (600, 0), (600, 100), (500, 100)],
        tower_roi=[(200, 200), (300, 200), (300, 300), (200, 300)],
        cube_colors=["blue", "green", "purple"], rod_colors=["red"], ball_colors=["orange", "green_ball"],
        avoid_balls=["white"], cable_colors=["cable"], px_per_cm=4.3)
    c.colors = {k: ColorRange(k, 0, 1, 0, 1, 0, 1) for k in
                ["blue", "green", "purple", "red", "orange", "green_ball", "white", "black", "cable"]}
    return c


def o(kind: str, color: str, x: float, y: float, bright: float = 100.0, elong: float = 1.0) -> Obj:
    return Obj(kind, color, x, y, 300, elong, (int(x) - 5, int(y) - 5, 10, 10), bright)


class FakeStation:
    """A table of symbolic objects and a pretend Dream-1: each new instruction closes the gripper, then (if it can)
    moves the object it names and opens again. `fail` lists instruction fragments that time out once."""

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.objs = [
            o("plug", "cable", 50, 50), o("plug", "cable", 70, 50),
            o("cube", "blue", *TOWER, bright=60), o("cube", "green", 380, 300, bright=150),
            o("cube", "purple", 400, 200, bright=110),
            o("rod", "red", 420, 330, elong=5.0),
            o("ball", "orange", 450, 150), o("ball", "orange", 470, 200), o("ball", "green_ball", 420, 120),
            o("avoid_ball", "white", 300, 60), o("eight_ball", "black", 200, 40),
        ]
        self.grip = {"left": 1.0, "right": 1.0}
        self.current = PARK
        self.polls = 0
        self.fail = list(fail)
        self.tasks: list[str] = []
        self.lock = threading.Lock()

    def set_task(self, text: str) -> None:
        with self.lock:
            self.current, self.polls = text, 0
            self.tasks.append(text)

    def task(self) -> str:
        return self.current

    def snapshot(self) -> Snapshot:
        with self.lock:
            self.polls += 1
            text = self.current
            failing = any(f in text for f in self.fail)
            if text != PARK and not failing:
                if self.polls == 2:
                    self.grip["right"] = 0.1
                elif self.polls == 4:
                    self._act(text)
                    self.grip["right"] = 1.0
            return Snapshot(np.zeros((360, 640, 3), np.uint8), 0.0, dict(self.grip))

    def world(self, c: Calibration) -> World:
        with self.lock:
            plugs = [p for p in self.objs if p.kind == "plug"]
            return World(list(self.objs), c, 0.0, cable_px_in_hub=40 * len(plugs))

    def _find(self, kind: str, color: str) -> Obj | None:
        return next((x for x in self.objs if x.kind == kind and x.color == color), None)

    def _act(self, text: str) -> None:
        if "USB cable out" in text:
            self.objs.remove(next(p for p in self.objs if p.kind == "plug"))
        elif m := re.search(r"(?:place|pick up) the (\w+) cube", text):
            cube = self._find("cube", m.group(1))
            cube.x, cube.y = TOWER
            rod = next((r for r in self.objs if r.kind == "rod" and getattr(r, "_in", None) == cube.color), None)
            if rod:
                rod.x, rod.y = TOWER
        elif m := re.search(r"put the (\w+) rod upright in the hole of the (\w+) cube", text):
            rod, cube = self._find("rod", m.group(1)), self._find("cube", m.group(2))
            rod.x, rod.y, rod.elongation = cube.x, cube.y, 1.0
            rod._in = cube.color  # moves with its cube
        elif m := re.search(r"pick up the (\w+) ping-pong ball and drop it in the bin", text):
            color = {"green": "green_ball"}.get(m.group(1), m.group(1))
            ball = next(b for b in self.objs if b.kind == "ball" and b.color == color and b.x < 500)
            ball.x, ball.y = BIN
        elif m := re.search(r"take the (\w+) ping-pong ball out of the bin", text):
            color = {"green": "green_ball"}.get(m.group(1), m.group(1))
            ball = next(b for b in self.objs if b.color == color and b.x >= 500)
            ball.x, ball.y = 450, 300


def make_server(fake: FakeStation) -> StationServer:
    c = cal()
    m = MatchConfig(own_color="orange", calibration=c, step_timeout_s=1.0, hold_s=0.05)
    srv = StationServer(fake, m, poll_s=0.01)
    def world():  # skip the camera: the fake's objects are the detections (one poll advances its pretend Dream-1)
        snap = fake.snapshot()
        return fake.world(c), snap.grip

    srv.world = world
    return srv


# ---------------------------------------------------------------------- server step logic
def test_steps_in_order() -> None:
    fake = FakeStation()
    srv = make_server(fake)
    srv.start("orange")
    assert srv.step("unplug")["ok"]
    assert srv.step("base", "blue")["ok"]
    assert srv.step("rod_in_cube", "red", "purple")["ok"]  # the rod in the top cube while it is on the table
    assert srv.step("place_cube", "green")["ok"]
    out = srv.step("place_cube", "purple")
    assert out["ok"] and "with the red rod standing in it" in out["instruction"]
    assert out["world"]["tower"]["height_in"] == pytest.approx(3 * 1.3 + 2.7)
    assert srv.step("hold")["ok"]
    out = srv.step("ball", "orange")
    assert out["ok"] and out["world"]["bin"]["own"] == 1
    # every step hands control back to "park" when it ends
    assert fake.tasks[-1] == PARK


def test_never_picks_white_or_black_and_hold_needs_height() -> None:
    srv = make_server(FakeStation())
    srv.start("orange")
    with pytest.raises(ValueError):
        srv._plan("ball", "white", "", srv.world()[0])
    with pytest.raises(ValueError):
        srv._plan("ball", "black", "", srv.world()[0])
    srv.step("base", "blue")
    out = srv.step("hold")
    assert not out["ok"] and "must reach" in out["message"]


def test_step_times_out_then_parks() -> None:
    fake = FakeStation(fail=("USB cable",))
    srv = make_server(fake)
    srv.start("orange")
    out = srv.step("unplug")
    assert not out["ok"] and "timed out" in out["message"]
    assert fake.task() == PARK


def test_ball_next_to_black_is_skipped() -> None:
    fake = FakeStation()
    fake.objs = [b for b in fake.objs if b.kind != "ball"] + [o("ball", "orange", 205, 45)]  # beside the 8-ball
    srv = make_server(fake)
    srv.start("orange")
    out = srv.step("ball", "orange")
    assert not out["ok"] and "clear of the black" in out["message"]


# ---------------------------------------------------------------------- HTTP + DimOS client + referee
def serve_fake(fake: FakeStation, port: int = 0):
    srv = make_server(fake)
    http = make_http(srv, "127.0.0.1", port)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    return http, f"http://127.0.0.1:{http.server_address[1]}"


def test_dimos_client_full_match() -> None:
    pytest.importorskip("requests")
    from dimos_hillclimb.rules import MatchRules, Refused
    from dimos_hillclimb.station_backend import DreamStation

    http, url = serve_fake(FakeStation())
    try:
        st = DreamStation(MatchRules(own_color="orange"), url)
        assert st.start()["ok"]
        with pytest.raises(Refused):  # a later stage's objects before unplug passes
            st.ball_to_bin("orange")
        assert st.unplug_cable()["ok"] and "unplug" in st.rules.passed
        with pytest.raises(Refused):
            st.stack_cube("green", "rod:red")  # the sim's tower shape is refused on the station
        assert st.stack_cube("blue", "tower_spot")["ok"]
        assert st.insert_rod("red", "purple")["ok"]
        assert not st.stack_cube("green", "purple")["ok"]  # purple isn't on the tower yet
        assert st.stack_cube("green", "blue")["ok"]
        assert st.stack_cube("purple", "green")["ok"]
        assert st.check_tower()["ok"] and "stack" in st.rules.passed
        with pytest.raises(Refused):
            st.ball_to_bin("white")
        with pytest.raises(Refused):
            st.ball_to_bin("black")
        assert st.ball_to_bin("orange")["ok"]
        assert st.ball_to_bin("orange")["ok"]
        assert st.rules.score() == 10 + 20 + 10
        scene = st.scene()
        assert scene["bin"]["own"] == 2 and scene["tower"]["height_in"] == pytest.approx(6.6)
        assert st.overhead_rgb().shape == (360, 640, 3)
    finally:
        http.shutdown()


if __name__ == "__main__":
    import argparse
    import logging

    ap = argparse.ArgumentParser(description="serve the fake station (no robot) for DimOS's station backend")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    fake = FakeStation()
    srv = make_server(fake)
    srv.m.step_timeout_s, srv.m.hold_s, srv.poll_s = 5.0, 0.5, 0.05
    print(f"fake station on http://127.0.0.1:{a.port}")
    make_http(srv, "127.0.0.1", a.port).serve_forever()
