"""Background poller for the station's GET /status, rendered as one line for every LLM turn.

Read-only, so it also runs in dry run. A poll never raises and never blocks a turn: failures become an
"unreachable" report, and a report older than STALE_AFTER_S reads as "no fresh status".
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from robot_gate import HOLD_FAILED_TEXT, PENDING_TEXT, SENT_TEXT, STOP_FAILED_SPEECH, MatchMode, hold_unconfirmed
from speech_guard import TRIGGER_STEM_RE
from station_client import StationClient, StationError

logger = logging.getLogger("station_monitor")

POLL_INTERVAL_S = 3.0
STALE_AFTER_S = 15.0
POLL_TIMEOUT_S = 6.0  # a little above StationClient's own 5 s status timeout


@dataclass(frozen=True)
class StationReport:
    fetched_at: float  # monotonic time of this poll
    reachable: bool
    status: dict[str, Any]  # the /status body; empty when unreachable
    error: str = ""  # why the poll failed


class StationMonitor:
    def __init__(
        self,
        client: StationClient,
        interval_s: float = POLL_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.interval_s = interval_s
        self._clock = clock
        self.report: StationReport | None = None
        self._last_ok: float | None = None
        self._task: asyncio.Task | None = None

    async def poll_once(self) -> None:
        """Fetch /status once and keep the result. Never raises (cancellation excepted)."""
        try:
            status = await asyncio.wait_for(self.client.status(), POLL_TIMEOUT_S)
            self.report = StationReport(self._clock(), True, status)
            self._last_ok = self.report.fetched_at
        except StationError as exc:
            self.report = StationReport(self._clock(), False, {}, exc.message)
        except Exception as exc:  # noqa: BLE001 - includes timeouts; a poll must never reach the session
            self.report = StationReport(self._clock(), False, {}, repr(exc))

    async def _run(self) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="station-monitor")

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def describe(self) -> str:
        """The status line for the LLM. Reads the newest report only; no I/O."""
        r = self.report
        if r is None:
            return "Station status: no fresh status yet (first poll has not finished)."
        age = self._clock() - r.fetched_at
        if age > STALE_AFTER_S:
            return f"Station status: no fresh status (last poll {age:.0f} s ago)."
        if not r.reachable:
            since = "" if self._last_ok is None else f", last answered {self._clock() - self._last_ok:.0f} s ago"
            return f"Station status (fetched {age:.0f} s ago): UNREACHABLE ({r.error}{since})."
        s = r.status
        parts = [str(s.get("state") or "unknown")]
        if s.get("prompt"):
            parts.append(f"task '{s['prompt']}'")
        if s.get("checkpoint"):
            parts.append(f"checkpoint {s['checkpoint']}")
        if s.get("state") in ("starting", "running", "stopping") and s.get("run_elapsed_s") is not None:
            parts.append(f"{s['run_elapsed_s']:.0f} s")
        err = s.get("last_error")
        if isinstance(err, dict):
            err = err.get("message") or str(err)
        if err:
            parts.append(f"last error: {err}")
        return f"Station status (fetched {age:.0f} s ago): " + " · ".join(parts)


HOLDING_LINE = "Holding."
FROZEN_REMINDER_S = 30.0
FROZEN_REMINDER = "The arms are holding. When the referee has scored, say end the run, then yes."
ALARM_SPEECH = "An arm raised an alarm and may drop. Please support it."
UNTRACKED_SPEECH = (
    "Another harness process is running on the station; I can't stop it by voice."
)
_ACTIVE_STATES = ("starting", "running", "freezing", "frozen", "homing", "parking")
_NO_TRIGGER_RE = TRIGGER_STEM_RE
FROZEN_REMINDER_SHORT = "Still holding."
FREEZING_TIMEOUT_S = 8.0  # status unreadable (or no live freezing seen) this long after a pending reply: escalate
HEARTBEAT_STALE_S = 3.0  # a freezing harness beats every 1 s; a reported heartbeat older than this is wedged
FREEZING_MAX_S = 30.0  # freezing for longer than this since freezing_since: escalate even with a live heartbeat
EIGHT_BALL_SPEECH = "The black eight ball is in our bin."
_REASON_WORDS = {
    "frozen_before_open": "the run was held before the arms opened, so nothing ever moved",
    "ended_before_open": "the run was ended before the arms opened, so nothing ever moved",
    "freeze_during_park": "a hold arrived during the park move, which cannot be stopped",
}


def _plain(text: object) -> str:
    """Status text for the LLM, with the trigger words rewritten (the model must never repeat them)."""
    return _NO_TRIGGER_RE.sub("hold", str(text))


class HarnessMonitor(StationMonitor):
    """StationMonitor for the relay harness: polls HarnessClient.status() and speaks reminders and alarms.

    `say` (set by the agent) is called with a fixed line, never with station text. Events come from fresh polls only,
    inside the polling task, so aclose() cancels them too. HarnessClient.status() and stop() share no lock (each call
    is its own subprocess), so a poll in flight never delays a freeze.
    """

    def __init__(self, client, interval_s: float = POLL_INTERVAL_S, clock: Callable[[], float] = time.monotonic,
                 say: Callable[[str], None] | None = None, reminder_s: float = FROZEN_REMINDER_S,
                 match: MatchMode | None = None, freezing_timeout_s: float = FREEZING_TIMEOUT_S,
                 heartbeat_stale_s: float = HEARTBEAT_STALE_S, freezing_max_s: float = FREEZING_MAX_S,
                 watch_tick_s: float = 0.5) -> None:
        super().__init__(client, interval_s, clock)
        self.heartbeat_stale_s = heartbeat_stale_s
        self.freezing_max_s = freezing_max_s
        self.watch_tick_s = watch_tick_s
        self._pending_armed: float = 0.0  # when the pending reply arrived
        self._live_freezing: float | None = None  # the last time a report showed freezing with a live heartbeat
        self.say = say
        self.reminder_s = reminder_s
        self.match = match if match is not None else MatchMode(enabled=False, clock=clock)
        self.freezing_timeout_s = freezing_timeout_s
        self._frozen_mark: float | None = None  # when the frozen state began or the last reminder was said
        self._reminded_full = False
        self._freezing_mark: float | None = None
        self._freezing_alarmed = False
        self._pending_task: asyncio.Task | None = None  # strong reference: the pending-freeze deadline timer
        self.retry_freeze: Callable[[], Awaitable[tuple[bool, str]]] | None = None  # the gate's freeze (with its retries)
        self._escalating = False
        self._retry_task: asyncio.Task | None = None
        self._eight_ball_spoken = False
        self._alarm_spoken = False
        self._untracked_spoken = False
        self._hold_failed_spoken = False

    async def poll_once(self) -> None:
        prev_ok = self._last_ok
        await super().poll_once()
        r = self.report
        if r is not None and r.reachable and r.status.get("ok") is False:
            # ok False: the helper could not report. Treat as unreachable, never as "holding".
            self.report = StationReport(r.fetched_at, False, {}, str(r.status.get("error") or "status failed"))
            self._last_ok = prev_ok
        self._events()

    # --- fail-safe for a pending freeze (the helper said "signalled, coming to a hold") ---
    def arm_pending_deadline(self) -> None:
        """Start a watchdog that does not depend on any poll. A pending freeze is not a fixed-time bet: `frozen` can
        be late for a good reason (a camera reopen, a Dream restart) while the harness beats a heartbeat. Unless a poll
        confirms the hold (or the gate settles it, or the session closes) first, it speaks STOP_FAILED_SPEECH once
        when no live `freezing` report has been seen for `freezing_timeout_s` (status unreadable or unreachable).
        Stale-heartbeat and over-long freezing are judged on each poll in `_events`."""
        if self._escalating or (self._pending_task is not None and not self._pending_task.done()):
            return  # the earliest deadline stands; a retry's own pending reply does not start another episode
        self._freezing_alarmed = False
        self._pending_armed = self._clock()
        self._live_freezing = None
        self._pending_task = asyncio.create_task(self._pending_deadline(), name="pending-freeze-deadline")

    async def _pending_deadline(self) -> None:
        while self._clock() - max(self._pending_armed, self._live_freezing or 0.0) <= self.freezing_timeout_s:
            await asyncio.sleep(min(self.watch_tick_s, self.freezing_timeout_s / 5))
        self._pending_task = None
        self._escalate()

    def _freezing_verdict(self, s: dict[str, Any], fetched_at: float) -> str:
        """For a report whose state is `freezing`: "live" (heartbeat fresh, not over-long), "bad" (stale heartbeat or
        freezing for too long), or "unknown" (an older harness that reports no heartbeat)."""
        beat = s.get("heartbeat_age_s")
        if not isinstance(beat, (int, float)):
            return "unknown"
        if beat > self.heartbeat_stale_s:
            return "bad"
        for_s = s.get("freezing_for_s")
        if isinstance(for_s, (int, float)) and for_s + max(0.0, self._clock() - fetched_at) > self.freezing_max_s:
            return "bad"
        return "live"

    def settle_pending(self) -> None:
        task, self._pending_task = self._pending_task, None
        if task is not None and not task.done():
            task.cancel()

    def _escalate(self) -> None:
        if not self._freezing_alarmed:
            if self._say(STOP_FAILED_SPEECH):  # done only once the line was handed to the speaker (which re-speaks once)
                self._freezing_alarmed = True
            self._start_retry()

    def _start_retry(self) -> None:
        """The hold was not confirmed in time: send the freeze again (the gate retries on fresh connections)."""
        if self.retry_freeze is None or (self._retry_task is not None and not self._retry_task.done()):
            return
        self._retry_task = asyncio.create_task(self._retry(), name="freeze-retry")

    async def _retry(self) -> None:
        self._escalating = True
        try:
            ok, text = await self.retry_freeze()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("freeze retry raised")
            ok, text = False, STOP_FAILED_SPEECH
        finally:
            self._escalating = False
        if not ok:
            self._say(text)
        elif text != PENDING_TEXT:
            self._say(HOLDING_LINE if text == SENT_TEXT else text)

    async def aclose(self) -> None:
        self.settle_pending()
        t, self._retry_task = self._retry_task, None
        if t is not None and not t.done():
            t.cancel()
        await super().aclose()

    def _say(self, text: str) -> bool:
        """Hand a line to the speaker (the agent's say_with_retry: re-speaks once if the human talks over it). True when
        it was accepted; callers mark an escalation done only on True, so a failed delivery is tried again."""
        if self.say is None:
            return False
        try:
            self.say(text)
            return True
        except Exception:  # noqa: BLE001 - a closed session must not kill the poller
            logger.debug("could not speak %r", text, exc_info=True)
            return False

    def fresh_status(self) -> dict[str, Any] | None:
        """The newest /status body if it is a fresh, reachable, ok report; else None."""
        r = self.report
        if r is None or not r.reachable or self._clock() - r.fetched_at > POLL_INTERVAL_S * 2:
            return None
        return r.status

    def _events(self) -> None:
        r = self.report
        if r is None or not r.reachable:
            return
        s = r.status
        state = s.get("state")
        if self.match.active:
            if state in ("starting", "running", "freezing", "frozen", "homing", "parking") and s.get("running") is not False:
                self.match.seen_active = True
            elif (state in ("done", "error", None) or s.get("running") is False) and self.match.may_end():
                self.match.stop()
        if s.get("eight_ball_alert"):
            if not self._eight_ball_spoken:
                self._eight_ball_spoken = self._say(EIGHT_BALL_SPEECH)  # an alarm: spoken even in match mode
        else:
            self._eight_ball_spoken = False
        hold_bad = state == "frozen" and s.get("running") is not False and hold_unconfirmed(s)
        if hold_bad:
            # `frozen` without a proven hold: never "Holding". Say so once (the alarm line below stays quiet), and the
            # pending-freeze deadline has nothing left to add.
            if not self._hold_failed_spoken and self._say(HOLD_FAILED_TEXT):
                self._hold_failed_spoken = True
                self._freezing_alarmed = True
                self._alarm_spoken = True
            if self._hold_failed_spoken:
                self.settle_pending()
        else:
            self._hold_failed_spoken = False
        if (state == "frozen" and s.get("running") is not False and not hold_bad) or (
                state == "done" and s.get("reason") in ("frozen_before_open", "ended_before_open")):
            self.settle_pending()  # the pending freeze is answered: the hold is confirmed, or nothing ever moved
        if state == "freezing" and s.get("running") is not False:
            now = self._clock()
            verdict = self._freezing_verdict(s, r.fetched_at)
            if verdict == "live":
                self._live_freezing = now
            elif verdict == "bad":
                self._escalate()
            elif self._freezing_mark is None:  # no heartbeat to judge by: the old fixed deadline
                self._freezing_mark = now
            elif now - self._freezing_mark > self.freezing_timeout_s:
                self._escalate()
        elif not hold_bad:
            self._freezing_mark = None
            self._freezing_alarmed = False
        if s.get("alarm"):
            if not self._alarm_spoken:
                self._alarm_spoken = self._say(ALARM_SPEECH)
        else:
            self._alarm_spoken = False
        if s.get("untracked_pids"):
            if not self._untracked_spoken:
                self._untracked_spoken = self._say(UNTRACKED_SPEECH)
        else:
            self._untracked_spoken = False
        if s.get("state") == "frozen" and s.get("running") is not False and not hold_bad:
            now = self._clock()
            if self._frozen_mark is None:
                self._frozen_mark = now
            elif now - self._frozen_mark >= self.reminder_s:
                self._frozen_mark = now
                if not self.match.active:  # match mode: no reminder at all
                    self._say(FROZEN_REMINDER_SHORT if self._reminded_full else FROZEN_REMINDER)
                    self._reminded_full = True
        else:
            self._frozen_mark = None
            self._reminded_full = False

    def describe(self) -> str:
        r = self.report
        if r is None:
            return "Station status: no fresh status yet (first poll has not finished)."
        age = self._clock() - r.fetched_at
        if age > STALE_AFTER_S:
            return f"Station status: no fresh status (last poll {age:.0f} s ago)."
        if not r.reachable:
            since = "" if self._last_ok is None else f", last answered {self._clock() - self._last_ok:.0f} s ago"
            return f"Station status (fetched {age:.0f} s ago): UNREACHABLE ({_plain(r.error)}{since}). Do not claim the arms are holding."
        s = r.status
        state = s.get("state")
        if s.get("running") is False and state in _ACTIVE_STATES:
            parts = [f"no hill climb process is running (the last state file says {"holding" if state == "frozen" else state}, so it probably died)"]
        elif state is None:
            parts = ["idle: no hill climb run"]
        else:
            parts = [self._state_words(s)]
        if s.get("stage") and state in ("running", "frozen"):
            parts.append(f"stage {s['stage']}")
        if s.get("dry_run"):
            parts.append("dry run, the arms are not live")
        if s.get("alarm"):
            parts.append(f"ALARM: {_plain(s['alarm'])}")
        if s.get("eight_ball_alert"):
            parts.append("ALERT: the black eight ball is in our bin")
        if s.get("holders"):
            parts.append(f"The arms are held by {_plain(', '.join(map(str, s['holders'])))} (not ours).")
        elif s.get("untracked_pids"):
            parts.append("another harness process is running on the station (not ours; Ctrl-C in its window ends it)")
        return f"Station status (harness hill climb, fetched {age:.0f} s ago): " + " · ".join(parts)

    @staticmethod
    def _state_words(s: dict[str, Any]) -> str:
        state = s.get("state")
        if state == "starting":
            return "starting: loading; the arms have not moved yet"
        if state == "running":
            if s.get("dry_run"):
                return "running (dry run): nothing moves"
            if not s.get("stage"):
                return "running: waiting at home for the hill climb to begin"
            return "running: the hill climb is driving the arms"
        if state == "frozen":
            if hold_unconfirmed(s):
                return "NOT confirmed holding: an arm did not take the hold and may still be moving"
            return "holding: the arms are holding in place"
        if state == "homing":
            return "homing: the arms are going home"
        if state == "parking":
            return "parking: the arms are moving to their park pose"
        if state == "freezing":
            return "coming to a hold: the signal was sent, not yet confirmed holding"
        if state == "done":
            score = s.get("score")
            text = "done" if score is None else f"done, score {score}"
            reason = _REASON_WORDS.get(s.get("reason"))
            return f"{text} ({reason})" if reason else text
        if state == "error":
            return f"error: {_plain(s.get('error') or 'unknown')}"
        return f"state {_plain(state)}"
