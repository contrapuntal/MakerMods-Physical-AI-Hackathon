"""The station server: the relay's steps as an HTTP API, for the DimOS harness (or any client) to call.

    ~/dream-env/bin/python -m harness serve --dry-run          # Dream-1 plans; nothing moves; steps time out
    ~/dream-env/bin/python -m harness serve                    # REAL: arms move on each /step. Hand on the e-stop.

It runs in the Dream library's Python, holds one Dream-1 control loop open for the match (station.py), and serves:

    GET  /health            {"ready": bool, "task": str}
    GET  /world             detections from the newest overhead frame, the tower so far, bin counts
    GET  /frame             the newest overhead frame as JPEG, detections drawn on
    POST /start             {"own_color": "orange"}: a new match: baselines (plugs at the hub, bin counts) reset
    POST /step              {"kind": ..., "target": ..., "on": ...}: one instruction to Dream-1, verified (below)
    POST /park              arms up and still

A step sets Dream-1's instruction (the relay's own wording, relay.py), polls the overhead camera every 0.5 s, and
passes once its check holds `confirm_polls` times in a row after the gripper has opened again; or it times out. Then
the instruction goes back to "park", so the model never keeps acting on its own between steps. Kinds:

    unplug                       a cable out of the hub
    base       target=cube       declares the cube the tower stands on (no motion; it stays where it is)
    place_cube target=cube       that cube onto the tower's top cube
    rod_in_cube target=rod on=cube   the rod upright in the cube's hole (on the table or on the tower)
    hold                         arms parked, the tower standing 3 s
    ball       target=colour     one ball of that colour into our bin
    take_out   target=colour     one ball of that colour out of our bin

The server knows nothing about stage order or scoring: the DimOS referee (dimos_hillclimb/rules.py) does that. It
does keep the relay's safety margins (a ball next to the black or a white ball is not picked).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import cv2

from harness import vision
from harness.config import MatchConfig, spoken
from harness.relay import AVOID, PARK
from harness.vision import World

_log = logging.getLogger(__name__)

POLL_S = 0.5
MAX_STEP_S = 60.0


class Station(Protocol):
    def set_task(self, text: str) -> None: ...
    def task(self) -> str: ...
    def snapshot(self): ...  # -> station.Snapshot | None


@dataclasses.dataclass
class StepResult:
    ok: bool
    message: str
    seconds: float = 0.0


class StationServer:
    """The step logic, independent of HTTP (tested with a fake station)."""

    def __init__(self, station: Station, match: MatchConfig, *, poll_s: float = POLL_S) -> None:
        self.station = station
        self.m = match
        self.cal = match.calibration
        self.poll_s = poll_s
        self.lock = threading.Lock()  # one step at a time
        self.reset(match.own_color)

    # ------------------------------------------------------------------ state
    def reset(self, own_color: str) -> None:
        self.m.own_color = own_color
        self.initial_plugs: int | None = None
        self.initial_cable_px = 0
        self.tower_xy: tuple[float, float] | None = None
        self.placed: list[str] = []  # "cube:blue", "rod:red", base first
        self.rod_cube: str | None = None
        self.rod_color: str | None = None

    def world(self) -> tuple[World, dict[str, float]] | None:
        snap = self.station.snapshot()
        if snap is None:
            return None
        rgb = cv2.resize(snap.rgb, self.cal.frame_size) if snap.rgb.shape[1::-1] != tuple(self.cal.frame_size) else snap.rgb
        return vision.detect(rgb, self.cal, snap.timestamp_s), snap.grip

    def start(self, own_color: str) -> dict:
        if own_color not in self.cal.ball_colors:
            by_spoken = {spoken(c): c for c in self.cal.ball_colors}
            if own_color not in by_spoken:
                raise ValueError(f"own_color must be one of {sorted(by_spoken)}")
            own_color = by_spoken[own_color]
        with self.lock:
            self.reset(own_color)
            seen = self.world()
            if seen is None:
                raise RuntimeError("no overhead frame yet")
            w = seen[0]
            self.initial_plugs, self.initial_cable_px = len(w.plugs_in_hub()), w.cable_px_in_hub
            self.station.set_task(PARK)
            return self.describe(w)

    def height_in(self) -> float:
        h = sum(p.startswith("cube:") for p in self.placed) * self.cal.cube_height_in
        if self.rod_cube and f"cube:{self.rod_cube}" in self.placed and f"rod:{self.rod_color}" in self.placed:
            h += self.cal.rod_length_in - self.cal.rod_insert_in
        return h

    def describe(self, w: World) -> dict:
        own, other = self.m.own_color, self.m.other_color()
        def obj(o):
            d = {"kind": o.kind, "color": spoken(o.color), "x_px": round(o.x), "y_px": round(o.y)}
            if o.kind == "rod":
                d["lying"] = o.lying
            if o.kind == "cube":
                d["brightness"] = round(o.brightness)  # darkest first is the relay's stacking order
            if o.kind in ("ball", "avoid_ball", "eight_ball"):
                d["in_bin"] = bool(w.inside(self.cal.bin_roi, [o]))
            return d
        return {
            "objects": [obj(o) for o in w.objects],
            "plugs_in_hub": len(w.plugs_in_hub()),
            "plugs_in_hub_at_start": self.initial_plugs,
            "tower": {"base_xy_px": self.tower_xy, "placed": list(self.placed), "height_in": round(self.height_in(), 2),
                      "pass_height_in": 6.0, "rod_in": self.rod_cube},
            "bin": {"own": len(w.balls_in_bin(own)), "other": len(w.balls_in_bin(other)),
                    "eight_ball": w.eight_ball_in_bin(), "white": len(w.avoid_in_bin())},
            "own_on_table": len(w.balls_on_table(own)),
            "own_color": spoken(own),
            "px_per_cm": self.cal.px_per_cm,
            "summary": w.summary(),
        }

    # ------------------------------------------------------------------ steps
    def step(self, kind: str, target: str = "", on: str = "") -> dict:
        with self.lock:
            seen = self.world()
            if seen is None:
                return {"ok": False, "message": "no overhead frame yet"}
            w0 = seen[0]
            try:
                plan = self._plan(kind, target, on, w0)
            except ValueError as exc:
                return {"ok": False, "message": str(exc), "world": self.describe(w0)}
            if isinstance(plan, StepResult):  # decided without moving
                return {"ok": plan.ok, "message": plan.message, "world": self.describe(w0)}
            instruction, check, needs_release, confirm, timeout_s, on_pass = plan
            result = self._run(instruction, check, needs_release, confirm, min(timeout_s, MAX_STEP_S))
            seen = self.world()
            w1 = seen[0] if seen else w0
            if result.ok and on_pass is not None:
                on_pass()
            return {"ok": result.ok, "message": result.message, "instruction": instruction,
                    "seconds": round(result.seconds, 1), "world": self.describe(w1)}

    def park(self) -> dict:
        self.station.set_task(PARK)
        return {"ok": True, "message": "parking: " + PARK}

    def _run(self, instruction: str, check, needs_release: bool, confirm: int, timeout_s: float) -> StepResult:
        self.station.set_task(instruction)
        t0 = time.monotonic()
        good = 0
        was_closed = released = False
        try:
            while time.monotonic() - t0 < timeout_s:
                time.sleep(self.poll_s)
                seen = self.world()
                if seen is None:
                    continue
                w, grip = seen
                closed = min(grip.values(), default=1.0) < 0.6
                if closed:
                    was_closed = True
                elif was_closed:
                    released = True
                ok = (released or not needs_release) and not closed and check(w)
                good = good + 1 if ok else 0
                if good >= confirm:
                    return StepResult(True, "done", time.monotonic() - t0)
            return StepResult(False, f"timed out after {timeout_s:.0f} s", time.monotonic() - t0)
        finally:
            self.station.set_task(PARK)

    def _plan(self, kind: str, target: str, on: str, w: World):
        cal, m = self.cal, self.m
        tol = 3.0 * cal.px_per_cm
        if kind == "unplug":
            if self.initial_plugs is None:
                raise ValueError("call /start first")
            def check(w: World) -> bool:
                fewer = self.initial_plugs > 0 and len(w.plugs_in_hub()) < self.initial_plugs
                fewer_px = self.initial_cable_px >= 15 and w.cable_px_in_hub < 0.6 * self.initial_cable_px
                return fewer or fewer_px
            return (f"pull one orange USB cable out of the white USB hub {AVOID['unplug']}", check, True, 3,
                    m.step_timeout_s, None)

        if kind == "base":
            cubes = w.of("cube", target)
            if not cubes:
                raise ValueError(f"no {target} cube in view")
            self.tower_xy = (cubes[0].x, cubes[0].y)
            self.placed = [f"cube:{target}"]
            self.rod_cube = self.rod_color = None
            return StepResult(True, f"the {target} cube at ({cubes[0].x:.0f}, {cubes[0].y:.0f}) px is the tower base")

        if kind == "place_cube":
            if self.tower_xy is None:
                raise ValueError("no tower base yet: declare one with kind=base")
            if f"cube:{target}" in self.placed:
                return StepResult(True, f"the {target} cube is already on the tower")
            below = next(p.split(":")[1] for p in reversed(self.placed) if p.startswith("cube:"))
            if target == self.rod_cube and f"rod:{self.rod_color}" in self.placed:
                text = (f"pick up the {target} cube with the {self.rod_color} rod standing in it by the sides of the "
                        f"cube and place it on top of the {below} cube {AVOID['stack']}")
            else:
                text = f"place the {target} cube on top of the {below} cube {AVOID['stack']}"
            def check(w: World) -> bool:
                return any(abs(c.x - self.tower_xy[0]) + abs(c.y - self.tower_xy[1]) < tol for c in w.of("cube", target))
            return text, check, True, 3, m.step_timeout_s, lambda: self.placed.append(f"cube:{target}")

        if kind == "rod_in_cube":
            if not on:
                raise ValueError("rod_in_cube needs on=<cube colour>")
            where = " on top of the tower" if f"cube:{on}" in self.placed else ""
            text = f"put the {target} rod upright in the hole of the {on} cube{where} {AVOID['stack']}"
            def check(w: World) -> bool:
                cubes, rods = w.of("cube", on), w.of("rod", target)
                if not cubes or any(r.lying for r in rods):
                    return False
                return not rods or any(r.dist_cm(c, cal.px_per_cm) < 4.0 for r in rods for c in cubes)
            def done() -> None:
                self.rod_cube, self.rod_color = on, target
                self.placed.append(f"rod:{target}")
            return text, check, True, 3, m.step_timeout_s, done

        if kind == "hold":
            if self.tower_xy is None:
                raise ValueError("no tower yet")
            if self.height_in() < m.min_height_in:
                return StepResult(False, f"the tower is {self.height_in():.1f} in; it must reach {m.min_height_in} in "
                                         "before the hold check")
            top = next(p.split(":")[1] for p in reversed(self.placed) if p.startswith("cube:"))
            since = [None]
            def check(w: World) -> bool:
                standing = any(abs(c.x - self.tower_xy[0]) + abs(c.y - self.tower_xy[1]) < 4.0 * cal.px_per_cm
                               for c in w.of("cube", top))
                now = time.monotonic()
                if not standing:
                    since[0] = None
                    return False
                since[0] = since[0] or now
                return now - since[0] >= m.hold_s
            return PARK, check, False, 1, m.hold_s + 6.0, None

        own, other = m.own_color, m.other_color()
        if kind == "ball":
            color = self._calibrated(target)
            if color in cal.avoid_balls or color == cal.eight_ball:
                raise ValueError(f"{target} balls are never picked")
            eight, avoid = w.of("eight_ball"), w.of("avoid_ball")
            clear = [b for b in w.balls_on_table(color) if w.nearest_cm(b, eight) >= m.clear_of_eight_cm
                     and w.nearest_cm(b, avoid) >= m.clear_of_avoid_cm]
            if not clear:
                return StepResult(False, f"no {target} ball on the table is clear of the black and white balls")
            start = len(w.balls_in_bin(color))
            text = f"pick up the {spoken(color)} ping-pong ball and drop it in the bin {AVOID['balls']}"
            return text, (lambda w: len(w.balls_in_bin(color)) > start), True, 3, m.step_timeout_s, None

        if kind == "take_out":
            if target in ("black", cal.eight_ball):
                return ("take the black ball out of the bin and put it on the table",
                        lambda w: not w.eight_ball_in_bin(), True, 3, m.step_timeout_s, None)
            color = self._calibrated(target)
            if color in cal.avoid_balls:
                return (f"take the {spoken(color)} ping-pong ball out of the bin and put it on the table",
                        lambda w: not w.inside(cal.bin_roi, w.of("avoid_ball", color)), True, 3, m.step_timeout_s,
                        None)
            start = len(w.balls_in_bin(color))
            if not start:
                return StepResult(True, f"no {target} ball is in the bin")
            return (f"take the {spoken(color)} ping-pong ball out of the bin and put it on the table",
                    lambda w: len(w.balls_in_bin(color)) < start, True, 3, m.step_timeout_s, None)

        raise ValueError(f"unknown step kind {kind!r}")

    def _calibrated(self, color: str) -> str:
        """`green` means the green_ball calibration, as on the harness command line."""
        names = self.cal.ball_colors + self.cal.avoid_balls + [self.cal.eight_ball]
        by_spoken = {spoken(c): c for c in names}
        if color in names:
            return color
        if color in by_spoken:
            return by_spoken[color]
        raise ValueError(f"no ball colour {color!r}; known: {sorted(by_spoken)}")


# ---------------------------------------------------------------------- HTTP
def make_http(server: StationServer, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            _log.info("http %s", fmt % args)

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path == "/health":
                    snap = server.station.snapshot()
                    return self._json(200, {"ready": snap is not None, "task": server.station.task()})
                if self.path == "/world":
                    seen = server.world()
                    return self._json(200, server.describe(seen[0])) if seen else self._json(503, {"error": "no frame"})
                if self.path == "/cameras":
                    aug = getattr(server.station, "augmenter", None)
                    return self._json(200, {"cameras": getattr(server.station, "camera_names", list)(),
                                            "sent": sorted(aug.last_sent) if aug else [],
                                            "augment": ",".join(aug.modes) if aug else None})
                if self.path.startswith("/camera/"):
                    rgb = getattr(server.station, "camera_rgb", lambda n: None)(self.path[len("/camera/"):])
                    if rgb is None:
                        return self._json(404, {"error": "no such camera, or no frame yet"})
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    return self._send(200, jpg.tobytes(), "image/jpeg")
                if self.path.startswith("/sent/"):
                    aug = getattr(server.station, "augmenter", None)
                    sent = aug.last_sent.get(self.path[len("/sent/"):]) if aug else None
                    if sent is None:
                        return self._json(404, {"error": "nothing sent under that key (is --augment on?)"})
                    return self._send(200, base64.b64decode(sent[1]), "image/jpeg")
                if self.path == "/frame":
                    snap = server.station.snapshot()
                    if snap is None:
                        return self._json(503, {"error": "no frame"})
                    seen = server.world()
                    rgb = vision.draw(cv2.resize(snap.rgb, server.cal.frame_size), seen[0]) if seen else snap.rgb
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    return self._send(200, jpg.tobytes(), "image/jpeg")
                self._json(404, {"error": f"no route {self.path}"})
            except Exception as exc:  # noqa: BLE001
                _log.exception("GET %s", self.path)
                self._json(500, {"error": repr(exc)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/start":
                    return self._json(200, server.start(body.get("own_color", server.m.own_color)))
                if self.path == "/step":
                    return self._json(200, server.step(body.get("kind", ""), body.get("target", ""), body.get("on", "")))
                if self.path == "/park":
                    return self._json(200, server.park())
                self._json(404, {"error": f"no route {self.path}"})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                _log.exception("POST %s", self.path)
                self._json(500, {"error": repr(exc)})

    return ThreadingHTTPServer((host, port), Handler)
