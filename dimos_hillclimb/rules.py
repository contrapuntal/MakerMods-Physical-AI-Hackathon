"""The match referee, shared by every backend: stage order, what may be touched when, and the score.

The DimOS agent (an LLM) decides what to do next, but the rubric's hard rules are enforced here, in code, so a model
mistake can't cost a disqualification:

- Stages run in order (unplug, stack, balls). A skill that touches a later stage's objects is refused until the
  current stage's pass bar is met.
- Unplug passes when a cable is out (+10). Stack passes when the tower is tall enough and stands on its own for 3 s
  after the arms let go (+20). Balls: +5 per own-colour ball in our bin, -5 per other-colour ball, and the black
  8-ball in our bin loses the match. White balls are never grabbed.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

STAGES = ("unplug", "stack", "balls")
NEVER_GRAB = ("white",)
EIGHT_BALL = "black"


class Outcome(dict):
    """What a station action did: {"ok": bool, "message": str, ...extra measured values}."""

    def __init__(self, ok: bool, message: str, **extra: Any) -> None:
        super().__init__(ok=ok, message=message, **extra)


class Refused(RuntimeError):
    """The referee blocked a skill; the message says why, written for the agent."""


@dataclasses.dataclass
class MatchRules:
    own_color: str = "orange"
    ball_colors: tuple[str, ...] = ("orange", "green")
    time_limit_s: float = 300.0
    stop_margin_s: float = 5.0
    passed: dict[str, float] = dataclasses.field(default_factory=dict)
    own_in_bin: int = 0
    other_in_bin: int = 0
    eight_in_bin: bool = False
    started_s: float | None = None
    log: list[str] = dataclasses.field(default_factory=list)

    @property
    def other_color(self) -> str:
        return next((c for c in self.ball_colors if c != self.own_color), "other")

    @property
    def stage(self) -> str:
        return next((s for s in STAGES if s not in self.passed), "done")

    def start(self) -> None:
        self.passed.clear()
        self.own_in_bin = self.other_in_bin = 0
        self.eight_in_bin = False
        self.log.clear()
        self.started_s = time.monotonic()
        self.note(f"go: own colour {self.own_color}, {self.time_limit_s:.0f} s")

    def elapsed_s(self) -> float:
        return 0.0 if self.started_s is None else time.monotonic() - self.started_s

    def time_left_s(self) -> float:
        return self.time_limit_s - self.elapsed_s()

    def note(self, what: str, stage: str | None = None) -> None:
        self.log.append(f"[{self.elapsed_s():6.1f} s] {stage or self.stage}: {what}")

    def require(self, stage: str, *, color: str | None = None) -> None:
        """Raise Refused unless a skill for `stage` may run now."""
        if self.started_s is None:
            raise Refused("the match has not started; call start_match first")
        if self.time_left_s() < self.stop_margin_s:
            raise Refused(f"time is up ({self.elapsed_s():.0f} s of {self.time_limit_s:.0f} s); park the arms")
        if self.stage == "done" and stage != "balls":
            raise Refused("every stage has passed")
        if STAGES.index(stage) > STAGES.index(self.stage if self.stage != "done" else "balls"):
            raise Refused(f"the {self.stage} stage has not passed yet; touching {stage} objects now is a "
                          f"disqualification. Finish {self.stage} first.")
        if STAGES.index(stage) < STAGES.index(self.stage if self.stage != "done" else "balls"):
            raise Refused(f"the {stage} stage already passed; leave its objects alone")
        if color in NEVER_GRAB:
            raise Refused(f"{color} balls are never grabbed")

    def pass_stage(self, stage: str, why: str) -> None:
        if stage == self.stage and stage not in self.passed:
            self.passed[stage] = round(self.elapsed_s(), 1)
            self.note(f"PASSED ({why})", stage)  # label it with the stage that passed, not the one now current

    def set_bin(self, own: int, other: int, eight: bool) -> None:
        self.own_in_bin, self.other_in_bin, self.eight_in_bin = own, other, eight

    def score(self) -> int:
        pts = (10 if "unplug" in self.passed else 0) + (20 if "stack" in self.passed else 0)
        return pts + 5 * self.own_in_bin - 5 * self.other_in_bin

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "passed": dict(self.passed),
            "score": self.score(),
            "own_color": self.own_color,
            "bin": {"own": self.own_in_bin, "other": self.other_in_bin, "eight_ball": self.eight_in_bin},
            "loses_on_eight_ball": self.eight_in_bin,
            "elapsed_s": round(self.elapsed_s(), 1),
            "time_left_s": round(self.time_left_s(), 1),
        }
