"""The real-station backend for HillClimbStation: a client of the station server (harness/station_server.py).

The station server runs in the Dream library's Python on the station (`~/dream-env/bin/python -m harness serve`),
holds Dream-1's control loop open, and turns each step into a Dream-1 instruction plus an overhead-camera check.
This side keeps the referee (rules.py) in front of every motion, so the stage order and the never-grab balls are
enforced here before anything reaches the arms. The two run in different Pythons (DimOS's venv here, Dream's there),
and the arms never depend on the DimOS process staying up: the server parks them after every step.

The real tower differs from the sim's: cubes stack on cubes and one rod stands in the top cube. So:
    stack_cube(cube, "tower_spot")   declares the base cube where it lies (no motion)
    insert_rod(rod, cube)            the rod into a cube's hole, best while that cube is still on the table
    stack_cube(cube, <cube colour>)  the cube onto the tower (the colour is the cube now on top; it's checked)
    check_tower()                    parks, then the 3 s free-standing hold
"""

from __future__ import annotations

import json

import numpy as np
import requests

from dimos_hillclimb.rules import EIGHT_BALL, MatchRules, Outcome, Refused

DEFAULT_URL = "http://127.0.0.1:8765"
STEP_HTTP_TIMEOUT_S = 90.0  # the server caps a step at 60 s


class DreamStation:
    def __init__(self, rules: MatchRules, url: str = DEFAULT_URL) -> None:
        self.rules = rules
        self.url = url.rstrip("/")
        self.http = requests.Session()
        self.last_world: dict = {}

    # ------------------------------------------------------------------ http
    def _get(self, path: str, timeout: float = 10.0) -> requests.Response:
        r = self.http.get(self.url + path, timeout=timeout)
        r.raise_for_status()
        return r

    def _post(self, path: str, body: dict, timeout: float = STEP_HTTP_TIMEOUT_S) -> dict:
        r = self.http.post(self.url + path, json=body, timeout=timeout)
        if r.status_code == 400:
            raise Refused(r.json().get("error", r.text))
        r.raise_for_status()
        return r.json()

    def _step(self, what: str, kind: str, target: str = "", on: str = "") -> dict:
        out = self._post("/step", {"kind": kind, "target": target, "on": on})
        self.last_world = out.get("world", self.last_world)
        self.rules.note(f"{what}: {'ok' if out['ok'] else 'failed'} ({out['message']}, {out.get('seconds', 0)} s)")
        return out

    # ------------------------------------------------------------------ match and perception
    def start(self, own_color: str | None = None, seed: int | None = None) -> Outcome:
        if own_color:
            if own_color not in self.rules.ball_colors:
                raise Refused(f"own_color must be one of {self.rules.ball_colors}")
            self.rules.own_color = own_color
        health = self._get("/health").json()
        if not health.get("ready"):
            return Outcome(False, "the station server is up but has no camera frame or arm state yet")
        self.last_world = self._post("/start", {"own_color": self.rules.own_color}, timeout=15)
        self.rules.start()
        return Outcome(True, f"match started on the station; own colour {self.rules.own_color}",
                       scene=self.scene())

    def scene(self) -> dict:
        world = self._get("/world").json()
        self.last_world = world
        return {**world, "match": self.rules.summary(),
                "note": "pixel positions are in the 640x360 overhead frame; px_per_cm converts to the table"}

    def overhead_rgb(self) -> np.ndarray:
        import cv2
        jpg = np.frombuffer(self._get("/frame").content, np.uint8)
        return cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ stage 1
    def unplug_cable(self) -> Outcome:
        self.rules.require("unplug")
        out = self._step("unplug", "unplug")
        if out["ok"]:
            self.rules.pass_stage("unplug", "a cable is out")
            return Outcome(True, "a USB cable is out of the hub; stage unplug passed (+10)",
                           match=self.rules.summary())
        return Outcome(False, f"unplug: {out['message']}", world=out.get("world"))

    # ------------------------------------------------------------------ stage 2
    def stack_cube(self, cube: str, on: str) -> Outcome:
        self.rules.require("stack")
        if on.startswith("rod:"):
            raise Refused("on the station, cubes stack on cubes and the rod goes in the top cube: use "
                          "stack_cube(cube, <the cube on top>) and insert_rod(rod, cube)")
        if on == "tower_spot":
            out = self._step(f"base {cube}", "base", cube)
            return Outcome(out["ok"], out["message"], tower=out.get("world", {}).get("tower"))
        tower = (self.last_world or {}).get("tower", {})
        cubes_on = [p.split(":")[1] for p in tower.get("placed", []) if p.startswith("cube:")]
        if cubes_on and on != cubes_on[-1]:
            return Outcome(False, f"the cube on top of the tower is {cubes_on[-1]}, not {on}", tower=tower)
        out = self._step(f"place {cube}", "place_cube", cube)
        tower = out.get("world", {}).get("tower")
        msg = f"the {cube} cube is on the tower; {tower['height_in']} in" if out["ok"] and tower else out["message"]
        return Outcome(out["ok"], msg, tower=tower)

    def insert_rod(self, rod: str, cube: str) -> Outcome:
        self.rules.require("stack")
        out = self._step(f"rod {rod} in {cube}", "rod_in_cube", rod, cube)
        msg = f"the {rod} rod stands in the {cube} cube" if out["ok"] else out["message"]
        return Outcome(out["ok"], msg, tower=out.get("world", {}).get("tower"))

    def check_tower(self) -> Outcome:
        self.rules.require("stack")
        out = self._step("hold", "hold")
        tower = out.get("world", {}).get("tower", {})
        if out["ok"]:
            self.rules.pass_stage("stack", f"tower {tower.get('height_in')} in standing 3 s")
            return Outcome(True, f"stack passed (+20): tower {tower.get('height_in')} in standing on its own",
                           tower=tower, match=self.rules.summary())
        return Outcome(False, out["message"], tower=tower)

    # ------------------------------------------------------------------ stage 3
    def ball_to_bin(self, color: str) -> Outcome:
        self.rules.require("balls", color=color)
        if color == EIGHT_BALL:
            raise Refused("the black 8-ball in our bin loses the match; never pick it")
        out = self._step(f"ball {color}", "ball", color)
        return self._after_balls(out, f"{color} ball")

    def take_out_of_bin(self, color: str) -> Outcome:
        self.rules.require("balls")
        if color == self.rules.own_color:
            raise Refused("that is our colour; leave it in the bin")
        out = self._step(f"take out {color}", "take_out", color)
        return self._after_balls(out, f"take out {color}")

    def _after_balls(self, out: dict, what: str) -> Outcome:
        b = out.get("world", {}).get("bin")
        if b:
            self.rules.set_bin(b["own"], b["other"], bool(b["eight_ball"]))
        msg = f"{what}: {out['message']}" + (f"; bin {json.dumps(b)}" if b else "")
        return Outcome(out["ok"], msg, match=self.rules.summary())

    def park(self) -> Outcome:
        out = self._post("/park", {}, timeout=10)
        return Outcome(True, out["message"])
