"""HillClimbStation: the Hill Climb relay as DimOS skills.

Each `@skill` is one sub-task the DimOS agent can call over MCP. The agent breaks the match down and sequences the
calls; this module does the motion and then measures whether it worked, and the referee (`rules.MatchRules`) refuses
any call the rubric forbids (a later stage's objects, white balls, the black 8-ball).

Backends (`backend` config field):
- "sim": `SimStation`, the MuJoCo relay table, with scripted motions per skill. Runs anywhere.
- "station": the real NUCK-3 station driven by Dream-1, through the station server (station_backend.py).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CAP_MOVEMENT
from dimos.agents.skill_result import SkillResult
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

from dimos_hillclimb.rules import MatchRules, Refused

_log = logging.getLogger(__name__)


class HillClimbStationConfig(ModuleConfig):
    backend: str = "sim"
    own_color: str = "orange"  # our ball colour from the coin flip
    time_limit_s: float = 300.0
    seed: int | None = None  # sim layout seed (None: random)
    station_url: str = "http://127.0.0.1:8765"  # the station server (harness/station_server.py), station backend only
    dashboard_host: str = "127.0.0.1"  # 0.0.0.0 to open the match dashboard from another machine
    dashboard_port: int = 8770  # 0: no dashboard


class HillClimbStation(Module):
    config: HillClimbStationConfig

    @rpc
    def start(self) -> None:
        super().start()
        rules = MatchRules(own_color=self.config.own_color, time_limit_s=self.config.time_limit_s)
        if self.config.backend == "sim":
            from dimos_hillclimb.sim_station import SimStation
            self._station: Any = SimStation(rules, seed=self.config.seed)
        elif self.config.backend == "station":
            from dimos_hillclimb.station_backend import DreamStation
            self._station = DreamStation(rules, self.config.station_url)
        else:
            raise ValueError(f"backend must be 'sim' or 'station', not {self.config.backend!r}")
        from dimos_hillclimb.dashboard import CallLog, Dashboard
        self._calls = CallLog()
        self._dashboard = None
        if self.config.dashboard_port:
            self._dashboard = Dashboard(lambda: self._station, self._calls, self.config.backend,
                                        self.config.dashboard_host, self.config.dashboard_port)
            self._dashboard.start()
        _log.info("HillClimbStation ready (%s backend)", self.config.backend)

    @rpc
    def stop(self) -> None:
        if getattr(self, "_dashboard", None) is not None:
            self._dashboard.stop()
        super().stop()

    # ------------------------------------------------------------------ plumbing
    def _call(self, name: str, *args: Any) -> SkillResult:
        rules = self._station.rules
        entry = self._calls.begin(name, [a for a in args if a is not None])
        try:
            out = getattr(self._station, name)(*args)
        except Refused as exc:
            self._calls.end(entry, "refused", str(exc), rules.elapsed_s() if rules.started_s else None)
            return SkillResult(message=f"REFUSED: {exc}", metadata={"ok": False, "refused": True,
                                                                   "match": rules.summary()})
        except Exception as exc:
            self._calls.end(entry, "error", repr(exc), rules.elapsed_s() if rules.started_s else None)
            raise
        ok = bool(out.get("ok"))
        self._calls.end(entry, "ok" if ok else "failed", out["message"], rules.elapsed_s() if rules.started_s else None)
        meta = {k: v for k, v in out.items() if k != "message"}
        return SkillResult(message=("OK: " if ok else "FAILED: ") + out["message"], metadata=meta)

    # ------------------------------------------------------------------ match and perception
    @skill
    def start_match(self, own_color: str = "", seed: int = -1) -> SkillResult:
        """Resets the table and starts the match clock. Call this once, first.

        Args:
            own_color: Our ping-pong ball colour from the coin flip, "orange" or "green". Empty keeps the configured one.
            seed: Sim only: the table layout seed, or -1 for the configured one.
        """
        return self._call("start", own_color or None, None if seed < 0 else seed)

    @skill
    def look(self) -> SkillResult:
        """Reports the table: the USB cable, every cube and rod with its position (metres) and whether it is on the
        tower, the tower's pieces and height, the balls on the table, the basket, and the match state (stage, score,
        time left). Does not move anything. Call it before planning and after any failure."""
        scene = self._station.scene()
        return SkillResult(message=json.dumps(scene), metadata={"ok": True})

    @skill
    def camera(self) -> Image:
        """Returns the overhead camera frame, for checking the table visually. Does not move anything."""
        return Image.from_numpy(self._station.overhead_rgb(), format=ImageFormat.RGB, frame_id="overhead")

    @skill
    def match_status(self) -> SkillResult:
        """Reports the stage, the stages passed, the score, the bin contents and the time left."""
        rules = self._station.rules
        return SkillResult(message=json.dumps({**rules.summary(), "log": rules.log[-12:]}), metadata={"ok": True})

    # ------------------------------------------------------------------ stage 1: unplug
    @skill(uses=[CAP_MOVEMENT])
    def unplug_cable(self) -> SkillResult:
        """Stage 1. Pulls one USB cable out of the hub and puts it down clear of the socket. The stage passes (+10)
        as soon as one cable is out."""
        return self._call("unplug_cable")

    # ------------------------------------------------------------------ stage 2: stack
    @skill(uses=[CAP_MOVEMENT])
    def stack_cube(self, cube: str, on: str) -> SkillResult:
        """Stage 2. Picks up a cube and puts it on the tower.

        Args:
            cube: The cube's colour: "blue", "green" or "purple".
            on: Where it goes: "tower_spot" for the first (base) cube, "rod:<colour>" to slide the cube's bottom
                hole onto a rod standing in the tower (e.g. "rod:red"; sim only), or a cube colour to sit it on that
                cube. On the station, "tower_spot" marks the cube as the base where it already lies (no motion).
        """
        return self._call("stack_cube", cube, on)

    @skill(uses=[CAP_MOVEMENT])
    def insert_rod(self, rod: str, cube: str) -> SkillResult:
        """Stage 2. Picks up a rod and stands it upright in the top hole of a cube (normally the tower's top cube).

        Args:
            rod: The rod's colour: "red", "green" or "yellow".
            cube: The cube whose hole receives it: "blue", "green" or "purple".
        """
        return self._call("insert_rod", rod, cube)

    @skill(uses=[CAP_MOVEMENT])
    def check_tower(self) -> SkillResult:
        """Stage 2 pass check. Parks both arms, waits 3 s, and passes the stage (+20) if the tower is at least
        6 in (0.152 m) tall and still standing on its own. Call it when you think the tower is tall enough."""
        return self._call("check_tower")

    # ------------------------------------------------------------------ stage 3: balls
    @skill(uses=[CAP_MOVEMENT])
    def ball_to_bin(self, color: str) -> SkillResult:
        """Stage 3. Picks one ball of this colour off the table and drops it in our basket (+5 for our colour,
        -5 for the other). White and black balls are refused: never try them.

        Args:
            color: The ball colour, normally our own colour.
        """
        return self._call("ball_to_bin", color)

    @skill(uses=[CAP_MOVEMENT])
    def take_out_of_bin(self, color: str) -> SkillResult:
        """Stage 3 recovery. Takes a ball that must not be in our basket (the other colour, a white one or the
        black 8-ball) out and puts it on the table.

        Args:
            color: The colour of the ball to remove.
        """
        return self._call("take_out_of_bin", color)

    @skill(uses=[CAP_MOVEMENT])
    def park_arms(self) -> SkillResult:
        """Moves both arms to rest with the grippers open. Safe at any time."""
        return self._call("park")
