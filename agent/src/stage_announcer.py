"""Announce the hill climb's stages by voice: "Picking up the cables now.", "Stacking the blocks now.", ...

Read-only and agent-side only. Every POLL_S it asks the station over ssh (the helper's ControlMaster, like
vision.fetch_station_ssh) for the instruction of the newest Dream round trip (the `prompt` of Dream's DebugView on
127.0.0.1:8093; no frames cross the network), maps it to a stage, and speaks one line when the stage changes. Retries of
the same stage say nothing. When the view goes away (the run ended) the last stage is forgotten, so the next run is
announced again. It never sends anything to the robot or the policy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable

import vision

logger = logging.getLogger("stage_announcer")

POLL_S = 3.0
GONE_AFTER_S = 8.0  # no reachable view for this long: the run is over, re-arm for the next one

LINES = {
    "unplug": "Picking up the cables now.",
    "stack": "Stacking the blocks now.",
    "balls": "Putting the golf balls into the basket now.",
    "eight_ball": "Taking the black ball back out of the bin.",
}

_PROMPT_SCRIPT = r'''
import json, sys, urllib.request
base, last = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
def get(path, timeout):
    return json.load(urllib.request.urlopen(base + path, timeout=timeout))
try:
    trips = get("/trips", 2)
except Exception as exc:
    sys.stderr.write(f"station view unreachable: {exc!r}"); sys.exit(3)
if not isinstance(trips, list) or not trips or not isinstance(trips[-1], dict) or "n" not in trips[-1]:
    sys.stderr.write("station view has no round trips yet"); sys.exit(4)
n = int(trips[-1]["n"])
if str(n) == last:
    sys.stdout.write(f"{n}\t="); sys.exit(0)  # same trip as last time: skip loading the whole trip (1 MB) on the station
try:
    obs = get(f"/trip/{n}", 3).get("observation") or {}
except Exception as exc:
    sys.stderr.write(f"station view unreachable: {exc!r}"); sys.exit(3)
prompt = obs.get("prompt") if isinstance(obs, dict) else None
sys.stdout.write(f"{n}\t{prompt if isinstance(prompt, str) else ''}")
'''

# Order matters: the black ball and leftover put-downs are checked before the generic ball/block words.
_RULES: list[tuple[str | None, re.Pattern]] = [
    ("eight_ball", re.compile(r"\bblack ball\b.*\bout of the bin\b|\btake the black ball\b", re.I)),
    (None, re.compile(r"\bput the .* down on the empty table\b|\bhold them still\b|\bhold the arms still\b", re.I)),
    ("unplug", re.compile(r"\bcables?\b|\bplugs?\b|\bport\b|\bpower strip\b", re.I)),
    ("stack", re.compile(r"\bstack\w*\b|\bblocks?\b|\bcubes?\b|\brods?\b|\btower\b", re.I)),
    ("balls", re.compile(r"\bballs?\b|\bbasket\b|\bbin\b", re.I)),
]


# The step-by-step instructions end with what NOT to touch ("... without touching the balls or the cables"): drop that
# clause first, or a stacking step would read as cables.
_AVOID = re.compile(r"[,;]?\s*\b(?:and\s+)?(?:without touching|do not touch|don'?t touch)\b.*$", re.I | re.S)


def classify(prompt: str | None) -> str | None:
    """The stage an instruction belongs to, or None (holds, put-downs, unknown text: nothing to announce)."""
    if not prompt:
        return None
    prompt = _AVOID.sub("", prompt)
    for stage, rx in _RULES:
        if rx.search(prompt):
            return stage
    return None


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get("STAGE_ANNOUNCE") or "1").strip().lower() not in ("0", "off", "no", "false")


class StageAnnouncer:
    def __init__(self, host: str, say: Callable[[str], object], clock=None, poll_s: float = POLL_S) -> None:
        self.host, self.say, self.poll_s = host, say, poll_s
        self._clock = clock or time.monotonic
        self.last_stage: str | None = None
        self.last_seen: float | None = None
        self._last_trip: tuple[int, str] | None = None  # (n, prompt) of the last trip read in full
        self._task: asyncio.Task | None = None

    async def fetch_prompt(self) -> tuple[int, str] | None:
        """(trip number, instruction) of the newest round trip, or None when no run serves the view."""
        last = str(self._last_trip[0]) if self._last_trip else ""
        argv = vision._ssh_frame_argv(self.host, (last,) if last else ())  # same ssh options and master
        try:
            rc, out, _err = await asyncio.wait_for(vision._run_ssh(argv, _PROMPT_SCRIPT), vision.STATION_SSH_TIMEOUT_S)
        except (TimeoutError, OSError):
            return None
        if rc != 0 or "\t" not in out:
            return None
        n_text, prompt = out.split("\t", 1)
        try:
            n = int(n_text)
        except ValueError:
            return None
        if prompt == "=" and self._last_trip and self._last_trip[0] == n:
            return self._last_trip  # unchanged since the last full read
        self._last_trip = (n, prompt)
        return self._last_trip

    async def step(self) -> str | None:
        """One poll: speaks and returns the line when the stage changed, else None."""
        now = self._clock()
        got = await self.fetch_prompt()
        if got is None:
            if self.last_seen is not None and now - self.last_seen >= GONE_AFTER_S:
                self.last_stage, self.last_seen = None, None  # run over: announce the next run from its first stage
            return None
        self.last_seen = now
        stage = classify(got[1])
        if stage is None or stage == self.last_stage:
            return None
        self.last_stage = stage
        line = LINES[stage]
        try:
            self.say(line)
        except Exception:
            logger.exception("could not announce %r", line)
        return line

    async def run(self) -> None:
        while True:
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("stage announcer poll failed")
            await asyncio.sleep(self.poll_s)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="stage-announcer")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
