"""Code-owned confirmation gate for robot motion.

The LLM may only *propose*. Execution needs a pending, unexpired action AND a fresh human turn:
the very next user message after the proposal, which must read as a yes. Single use, one user turn only.
stop is never gated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from speech_guard import scrub_text
from station_client import StationClient, StationError, StationResult

logger = logging.getLogger("robot_gate")

PENDING_TTL_S = 15.0
STOP_DEBOUNCE_S = 2.0

YES_RE = re.compile(
    r"^\W*(?:(?:uh|um|ok(?:ay)?|so|well)\W+)*(yes|yeah|yep|yup|sure|go ahead|do it|go for it|confirm(?:ed)?)\b"
)
NO_RE = re.compile(
    r"\b(no|don'?t|not|wait|stop|cancel|never ?mind|hold on|hang on|nah|actually|abort|later|maybe|nope|cut|freeze|halt|instead|different|other one|wrong|after|minute|second|first)\b"
)
# Replies shorter than this only count as an echo if they equal the first tokens of the agent's message.
ECHO_MIN_WORDS = 3


def normalize(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'").lower().strip()


def tokens(text: str) -> list[str]:
    """Lowercase word tokens with [..] / <..> expressive tags and punctuation removed."""
    t = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", normalize(text))
    return re.findall(r"[a-z0-9']+", t)


def _is_echo(user: list[str], agent: list[str]) -> bool:
    if not user or not agent:
        return False
    if len(user) < ECHO_MIN_WORDS:
        return user == agent[: len(user)]  # a short reply that just repeats how the agent's message began
    n = len(user)
    return any(agent[i : i + n] == user for i in range(len(agent) - n + 1))


_FILL = r"(?:uh|um|ok|okay|so|well|alright|all right)"
_ASSENT = r"(?:yes|yeah|yep|yup|sure|go ahead|do it|go for it|confirm|confirmed)"
_TAIL = r"(?:please|now|thanks|thank you|i'm sure|i am sure)"
# A complete utterance of unconditional assent: fillers, then assent words (with an optional please/thanks tail). Anything
# else (conditions, quotes, reports such as "he said yes", "say yes", "yes only when I am clear") is not a yes.
ASSENT_RE = re.compile(rf"^(?:{_FILL} )*{_ASSENT}(?: (?:{_ASSENT}|{_TAIL}|{_FILL}))*$")
ASSENT_PREFIX_RE = re.compile(rf"^(?:{_FILL} )*{_ASSENT}(?: (?:{_ASSENT}|{_TAIL}|{_FILL}))*(?= |$)")
MAX_ASSENT_WORDS = 7
MAX_ECHO_WORDS = 10
_ECHO_EXTRA = frozenset({"it", "now", "please", "the", "thanks"})


def _restates_action(rest: list[str], action: str) -> bool:
    """'yes, end the run': the words after the assent only restate the PENDING action, starting with its verb.
    A different action ('yes, start the hill climb' while an end is pending) is not a yes."""
    words = tokens(action.split(":", 1)[0])  # "start the task: <prompt>" -> start the task
    return bool(rest and words) and rest[0] == words[0] and all(w in words or w in _ECHO_EXTRA for w in rest)


def is_clear_yes(text: str, last_assistant: str = "", action: str = "") -> bool:
    t = normalize(text)
    if not t or "?" in t:
        return False
    toks = tokens(t)
    if not toks or NO_RE.search(t):
        return False
    joined = " ".join(toks)
    if not (len(toks) <= MAX_ASSENT_WORDS and ASSENT_RE.match(joined)):
        m = ASSENT_PREFIX_RE.match(joined)
        if not (m and action and len(toks) <= MAX_ECHO_WORDS
                and _restates_action(joined[m.end():].split(), action)):
            return False
    return not _is_echo(toks, tokens(last_assistant))


# The only voice emergency words. "stop", "cut" and "wait" are NOT triggers (but "stop"/"cut" stay in NO_RE).
FREEZE_WORD_RE = re.compile(r"\b(freeze|halt)\b", re.IGNORECASE)
STOP_WORD_RE = FREEZE_WORD_RE  # alias for existing imports

# Agent-readable-aloud strings avoid the words freeze/halt (and never open with stop/cut) so the agent's own voice cannot trip the fast path.
SENT_TEXT = "Sent; the arms are holding."
STARTING_TEXT = "Accepted; the run stops before the arms move."
ABORTED_TEXT = "Done; the run ended before the arms moved."
PENDING_TEXT = "Sent; the arms are coming to a hold."
ALREADY_OVER_TEXT = "The run is already over: the arms are parked and switched off; nothing is moving."
RESEND_OK_TEXT = (
    "Already handled: an emergency hold arrived while this was in flight and was re-sent afterwards; the arms are holding."
)
RESEND_STARTING_TEXT = (
    "Already handled: an emergency hold arrived while this was in flight and was re-sent afterwards; the run stops before "
    "the arms move."
)
HOLD_FAILED_TEXT = "Not confirmed: an arm did not take the hold and may still be moving."
FAIL_REPEAT_S = 1.0  # after a failed fast stop, a repeat of the same utterance (interim then final) is ignored
# Only merges the fast path, the tool and the re-send reporting the SAME failed freeze (they resolve within ms of each
# other). Kept short and reset by any confirmed freeze: a missed alarm for a new failure is worse than a duplicate.
ALARM_EPISODE_S = 2.0
# There is no physical e-stop on this setup: a freeze that is not confirmed is the SYSTEM's to act on (it retries) and
# say plainly. These lines state what is true and never tell the human to press anything.
ESTOP_NOW = "Not confirmed: the arms may still be moving."
STOP_FAILED_SPEECH = ESTOP_NOW
UNREACHABLE_TEXT = "I could not reach the station; the arms may still be moving."
NOTHING_RUNNING_TEXT = "Nothing is running; the arms aren't moving."
RETRYING_SPEECH = "Not confirmed yet; trying again."
NOT_OURS_PREFIX = "Another program"
RETRY_ATTEMPTS = 2  # fresh-connection retries after the first freeze fails
RETRY_TOTAL_S = 2.5  # all retries together; with the 2 s connect timeout of the first try the human hears the outcome in <= ~4.6 s
RECHECK_S = 1.0  # one status re-check after a failed attempt
RETRY_EPISODE_S = 10.0  # the "trying again" line is said once per episode
FAST_OWN_S = 15.0  # how long a fast-path freeze owns the user message that triggered it


def not_ours_text(holders) -> str:
    names = ", ".join(str(h) for h in holders or [] if h)
    who = f", {names}," if names else ""
    return f"Another program{who} is driving the arms; I can't stop it by voice."


def is_stop_failure(text: str) -> bool:
    """True for any line the gate returns when a freeze did not confirm (all are spoken as the alarm)."""
    return text in (ESTOP_NOW, UNREACHABLE_TEXT, HOLD_FAILED_TEXT) or text.startswith(NOT_OURS_PREFIX)


def hold_unconfirmed(d) -> bool:
    """True when a `frozen` harness record does not prove the hold: hold_ok False, or (older harness, no hold_ok) an arm
    alarm/fault field set. A later alarm on a run that wrote hold_ok True is a separate alarm, not a failed hold."""
    if not isinstance(d, dict) or d.get("state") != "frozen":
        return False
    if "hold_ok" in d and d["hold_ok"] is not None:
        return d["hold_ok"] is not True
    return bool(d.get("alarm") or d.get("fault"))
# A word right before the trigger that negates it ("don't freeze", "do not halt", "never freeze"). NOT "no":
# "No, freeze!" is a panicked command and must fire.
NEGATIONS = frozenset({"don't", "dont", "not", "never"})
MATCH_START_GRACE_S = 20.0


class MatchMode:
    """Match mode: the agent stays silent while a hill climb runs. On after a confirmed relay start; the monitor turns it
    off when the run is done, in error or gone. `enabled` False (MATCH_MODE=0, station backend) keeps it always off."""

    def __init__(self, enabled: bool = True, clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = enabled
        self._clock = clock
        self.active = False
        self.seen_active = False  # the monitor has seen the run itself since the start was confirmed
        self.started_at = -1e9

    def start(self) -> None:
        if self.enabled:
            self.active = True
            self.seen_active = False
            self.started_at = self._clock()

    def stop(self) -> None:
        self.active = False
        self.seen_active = False

    def may_end(self) -> bool:
        """The monitor may turn it off only once it saw the run, or after a grace period (a stale 'done' file)."""
        return self.seen_active or self._clock() - self.started_at > MATCH_START_GRACE_S


@dataclass(frozen=True)
class UserTurn:
    """The latest user chat message at tool time: its id, text, and ordinal (count of user messages)."""

    id: str
    text: str
    count: int
    last_assistant: str = ""  # most recent assistant message before this user turn (echo guard)


@dataclass
class PendingAction:
    kind: str  # "start" | "home" | "relay" | "end"
    prompt: str | None
    expires_at: float
    user_id: str
    user_count: int
    clock_restarted: bool = False  # the TTL restarts once, when the agent finishes asking

    def describe(self) -> str:
        if self.kind == "home":
            return "send the arms home"
        if self.kind == "relay":
            return "start the hill climb"
        if self.kind == "end":
            return "end the run"
        return f"start the task: {self.prompt}"


class RobotGate:
    def __init__(
        self,
        client: StationClient | None = None,
        ttl_s: float = PENDING_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client or StationClient()
        self.ttl_s = ttl_s
        self._clock = clock
        self.pending: PendingAction | None = None
        self._stop_epoch = 0
        self._last_fast_stop = -1e9
        self._tasks: set[asyncio.Task] = set()  # fast-stop tasks in flight (fast_stop skips while non-empty)
        self._bg: set[asyncio.Task] = set()  # strong references: shielded motion calls and tool-owned stops
        self._inflight = 0  # relay/start/home/end calls in flight
        self._fail_at = -1e9
        self._last_alarm = -1e9
        self.on_resend_failed: Callable[[], None] | None = None
        self.on_retry: Callable[[], None] | None = None  # speaks RETRYING_SPEECH when a freeze needs a retry
        self._last_retry_say = -1e9
        self._fast_turn: dict | None = None  # the last fast-path freeze: {"words", "at", "task", "claimed"}
        # Fail-safe for a pending freeze reply: on_pending arms a deadline, on_settled disarms it.
        self.on_pending: Callable[[], None] | None = None
        self.on_settled: Callable[[], None] | None = None
        self.match = MatchMode(enabled=False, clock=clock)
        # Optional: () -> the monitor's fresh /status dict, or None. Used so a dry end needs no extra ssh round trip.
        self.fresh_status: Callable[[], dict | None] | None = None

    def restart_pending_clock(self) -> None:
        """Restart the pending action's TTL once, when the agent has finished speaking its question.

        The proposal is created before the question is spoken, and speaking it can take most of the TTL; the human
        cannot answer before hearing it. Only once per proposal, so a forgotten proposal still expires, and the
        one-turn rule (the very next user message) still applies.
        """
        p = self.pending
        if p is not None and not p.clock_restarted:
            p.expires_at = self._clock() + self.ttl_s
            p.clock_restarted = True

    # --- proposals (never move anything; every call first clears any older pending action) ---
    def _propose(self, kind: str, prompt: str | None, turn: UserTurn | None) -> str:
        if turn is None:
            return "Refused: no user message to attach the request to."
        self.pending = PendingAction(kind, prompt, self._clock() + self.ttl_s, turn.id, turn.count)
        extra = ""
        if kind == "home":
            extra = (
                " Warn the human: once homing starts, voice commands do NOT work (the station refuses them "
                "during a home move), so the home move cannot be interrupted."
            )
        elif kind == "end":
            extra = (
                " Warn the human: ending the run sends the arms home and then parks them, and that move can knock over a "
                "standing tower. The emergency word still holds the arms during the home move, but it cannot stop "
                "the final park."
            )
        return (
            f"Pending, NOT yet done: {self.pending.describe()}. Ask the human now, reading the task back "
            f'exactly, word for word: "{self.pending.describe()}". Then wait for their spoken '
            f"answer; do not call confirm_pending_action in this reply.{extra}"
        )

    def propose_task(self, prompt: str, turn: UserTurn | None) -> str:
        self.pending = None
        if not os.environ.get("DREAM_CHECKPOINT"):
            return "Refused: no model checkpoint is configured (DREAM_CHECKPOINT unset). Tell the human; do not guess one."
        if not prompt or not prompt.strip():
            return "Refused: empty task."
        return self._propose("start", prompt.strip(), turn)

    async def request_task(self, prompt: str, turn: UserTurn | None) -> str:
        """A new task, whether or not a run is active. Never moves the arms.

        If the station says a run is active, end it first (the ungated stop path), then propose a gated start
        with the new task. A failed stop returns ESTOP_NOW and proposes nothing. If the status cannot be read,
        this is a plain proposal: the station refuses /start while a run exists.
        """
        self.pending = None  # an older proposal must not stay confirmable while we await the station
        if not await self._run_active():
            return self.propose_task(prompt, turn)
        if not os.environ.get("DREAM_CHECKPOINT") or not prompt or not prompt.strip():
            return self.propose_task(prompt, turn)  # refuses; the run is left alone
        ok, text = await self.stop_result()
        if not ok:
            return text
        return (
            "The running task was ended and the arms are holding. Tell the human that in plain words, "
            f"without naming the emergency word or saying stop or cut. {self.propose_task(prompt, turn)}"
        )

    async def _run_active(self) -> bool:
        try:
            state = (await self.client.status()).get("state")
        except Exception as exc:  # noqa: BLE001 - unreachable station: treat as no known run
            logger.warning("status check before task change failed: %r", exc)
            return False
        return state in ("starting", "running")

    def propose_home(self, turn: UserTurn | None) -> str:
        self.pending = None
        return self._propose("home", None, turn)

    def propose_relay_start(self, turn: UserTurn | None) -> str:
        self.pending = None
        return self._propose("relay", None, turn)

    def propose_end(self, turn: UserTurn | None) -> str:
        self.pending = None
        return self._propose("end", None, turn)

    def cancel(self) -> str:
        had = self.pending is not None
        self.pending = None
        return "Cancelled the pending request." if had else "Nothing was pending."

    # --- execution ---
    async def confirm(self, turn: UserTurn | None) -> str:
        action, self.pending = self.pending, None  # popped FIRST: any refusal also consumes it
        if action is None:
            return "Nothing is pending, so nothing was done. Ask what the human wants first."
        if self._clock() > action.expires_at:
            return (
                "The pending request expired, so nothing was done. Ask the human again if they still want it."
            )
        if turn is None or turn.id == action.user_id or turn.count != action.user_count + 1:
            return (
                "Refused: the human has not answered since the proposal. Nothing was done and the request is "
                "cleared. Ask them, wait for their answer, and propose again if needed."
            )
        if not is_clear_yes(turn.text, turn.last_assistant, action.describe()):
            return (
                "Refused: the human's reply was not a clear yes. Nothing was done and the request is cleared."
            )
        try:
            if action.kind == "relay":
                res, exc, restop = await self._guarded(self.client.start_relay)
            elif action.kind == "end":
                if not self.client.live and not await self._harness_known_dry():
                    return (
                        "Could not confirm the harness run is a dry run, so nothing was sent. If a run is live, "
                        "end it with Ctrl-C at the station."
                    )
                # The pending-freeze watchdog is settled only once the end was delivered (below); never before.
                res, exc, restop = await self._guarded(self.client.end)
                if restop is None and exc is None and not _data_failed(getattr(res, "data", None)):
                    self._settle_pending()
            elif action.kind == "start":
                checkpoint = os.environ.get("DREAM_CHECKPOINT", "")
                if not checkpoint:
                    return "Refused: no model checkpoint is configured."
                res, exc, restop = await self._guarded(
                    lambda: self.client.start(action.prompt or "", checkpoint)
                )
            else:
                res, exc, restop = await self._guarded(self.client.home)
        except Exception as exc2:  # noqa: BLE001 - never let a tool crash silently
            return f"The request failed unexpectedly: {exc2!r}."
        if restop is not None:
            ok, rtext = restop
            if not ok:
                return ESTOP_NOW
            prefix = ""
            if isinstance(exc, StationError):
                prefix = f"The station refused or failed: {exc.message}. "
            elif exc is not None:
                prefix = f"The request failed unexpectedly: {exc!r}. "
            elif action.kind in ("relay", "end") and _data_failed(getattr(res, "data", None)):
                prefix = f"The harness refused: {res.data.get('error') or 'unknown error'}. "
            return prefix + (RESEND_OK_TEXT if rtext == SENT_TEXT else RESEND_STARTING_TEXT)
        if isinstance(exc, StationError):
            return f"The station refused or failed: {exc.message}. Nothing else was done."
        if exc is not None:
            return f"The request failed unexpectedly: {exc!r}."
        if action.kind in ("relay", "end"):
            data = getattr(res, "data", None)
            if _data_failed(data):  # the helper's own refusal; a missing ok means success
                return f"The harness refused: {data.get('error') or 'unknown error'}. Nothing else was done."
        if action.kind == "relay" and not getattr(res, "dry_run", False):
            self.match.start()  # a confirmed relay start (the helper said ok, or sent no ok key)
        return _summarize(action, res, getattr(self.client, "live", True))

    async def _harness_known_dry(self) -> bool:
        """For an agent that is not live: True only if the status positively shows no live harness run."""
        st = None
        if self.fresh_status is not None:
            try:
                st = self.fresh_status()
            except Exception:  # noqa: BLE001
                st = None
        if st is None:
            try:
                st = await self.client.status()
            except Exception:  # noqa: BLE001
                return False
        if not isinstance(st, dict) or st.get("ok") is not True or st.get("untracked_pids"):
            return False
        return st.get("running") is False or st.get("dry_run") is True

    async def _guarded(self, fn):
        """Run a motion request as its own task (so an interrupted confirm cannot lose the re-send); if a stop
        happened while it was in flight, send the stop again afterwards.

        Returns (result, exception, restop) where restop is None (not re-sent) or the re-send's (ok, text).
        """
        self._inflight += 1  # counted and epoch-stamped now, not when the task first runs
        task = asyncio.create_task(self._guarded_inner(fn, self._stop_epoch))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return await asyncio.shield(task)

    async def _guarded_inner(self, fn, epoch: int):
        res = exc = restop = None
        try:
            res = await fn()
        except Exception as e:  # noqa: BLE001
            exc = e
        finally:
            self._inflight -= 1
        if self._stop_epoch != epoch:
            restop = await self._do_stop(strict=True)
            if not restop[0] and self.on_resend_failed is not None:
                try:
                    self.on_resend_failed()
                except Exception:  # noqa: BLE001
                    logger.exception("resend-failed callback raised")
        return res, exc, restop

    def _settle_pending(self) -> None:
        if self.on_settled is not None:
            try:
                self.on_settled()
            except Exception:  # noqa: BLE001
                logger.exception("pending-freeze hook raised")

    def alarm_due(self) -> bool:
        """True once per failed-freeze episode: the fast path, the tool and the re-send share one spoken alarm."""
        now = self._clock()
        if now - self._last_alarm < ALARM_EPISODE_S:
            return False
        self._last_alarm = now
        return True

    # --- stop (never gated) ---
    async def _do_stop(self, strict: bool = False) -> tuple[bool, str]:
        ok, text = await self._do_stop_inner(strict)
        # A pending reply arms the monitor's fail-safe deadline; any other answer (confirmed, or failed and alarmed
        # by the caller) settles it.
        hook = self.on_pending if ok and text == PENDING_TEXT else self.on_settled
        if hook is not None:
            try:
                hook()
            except Exception:  # noqa: BLE001
                logger.exception("pending-freeze hook raised")
        return ok, text

    def _verdict(self, res, strict: bool) -> tuple[bool, str] | None:
        """(ok, text) for a clear outcome, or None for a failure worth retrying."""
        data = getattr(res, "data", None)
        if isinstance(data, dict) and (data.get("hold_failed") or hold_unconfirmed(data)):
            logger.warning("freeze: the hold is not confirmed: %s", data.get("hold_failures") or data.get("error"))
            return False, HOLD_FAILED_TEXT  # never retried and never "Holding."
        if isinstance(data, dict) and (data.get("not_ours") or data.get("untracked_pids")) and data.get("ok") is not True:
            logger.warning("freeze: not ours: %s", data.get("error"))
            return False, not_ours_text(data.get("holders"))  # never retried: nothing of ours to stop
        if _data_failed(data):
            logger.warning("stop failed: %s", data.get("error"))
            return None if not data.get("final") else (False, ESTOP_NOW)
        if isinstance(data, dict) and data.get("nothing_running"):
            if strict:  # a start/end was in flight: "nothing is running" cannot be trusted yet
                logger.warning("freeze said nothing running while a start/end was in flight: unconfirmed")
                return None
            return True, NOTHING_RUNNING_TEXT
        if isinstance(data, dict) and data.get("already_over"):
            if strict:  # a start/end was in flight: "nothing is moving" cannot be trusted yet
                logger.warning("freeze said already over while a start/end was in flight: unconfirmed")
                return None
            return True, ALREADY_OVER_TEXT
        if isinstance(data, dict) and data.get("note") and data.get("state") == "done":
            return True, ABORTED_TEXT  # aborted before any arm opened: nothing moved, nothing to park
        if isinstance(data, dict) and data.get("note") and data.get("state") == "starting":
            return True, STARTING_TEXT
        if isinstance(data, dict) and data.get("pending") and data.get("state") == "freezing":
            return True, PENDING_TEXT  # signalled, the arms are coming to a hold; the monitor watches for the rest
        return True, SENT_TEXT

    async def _attempt(self, call, strict: bool) -> tuple[tuple[bool, str] | None, bool]:
        """One freeze attempt: (verdict or None to retry, was the failure a transport one)."""
        try:
            res = await call()
        except StationError as exc:
            if exc.refused and "already stopping" in exc.message:
                return (True, "Already in progress; the arms should hold in a moment."), False
            logger.warning("stop failed: %s", exc.message)
            return None, False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - includes timeouts and anything unexpected
            logger.warning("stop failed: %r", exc)
            return None, False
        verdict = self._verdict(res, strict)
        data = getattr(res, "data", None)
        return verdict, bool(isinstance(data, dict) and data.get("transport"))

    async def _recheck(self, timeout_s: float) -> tuple[bool, str] | None:
        """Re-read the state on a fresh connection after a failed attempt. A hold seen there confirms it."""
        fn = getattr(self.client, "status_fresh", None)
        if fn is None or timeout_s <= 0:
            return None
        try:
            st = await asyncio.wait_for(fn(), min(RECHECK_S, timeout_s))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(st, dict) or st.get("ok") is not True:
            return None
        if st.get("untracked_pids") or (st.get("holders") and st.get("running") is False):
            return False, not_ours_text(st.get("holders"))
        if st.get("state") == "frozen" and st.get("running") is not False:
            if hold_unconfirmed(st):
                return False, HOLD_FAILED_TEXT
            return True, SENT_TEXT
        return None

    def _say_retry(self) -> None:
        now = self._clock()
        if now - self._last_retry_say < RETRY_EPISODE_S or self.on_retry is None:
            return
        self._last_retry_say = now
        try:
            self.on_retry()
        except Exception:  # noqa: BLE001
            logger.exception("retry callback raised")

    async def _do_stop_inner(self, strict: bool = False) -> tuple[bool, str]:
        verdict, transport = await self._attempt(self.client.stop, strict)
        if verdict is not None:
            return verdict
        fresh = getattr(self.client, "stop_fresh", None)
        if fresh is None:  # a client with no second connection to try (the station backend)
            return False, UNREACHABLE_TEXT if transport else ESTOP_NOW
        self._say_retry()
        loop = asyncio.get_running_loop()
        end = loop.time() + RETRY_TOTAL_S
        for _ in range(RETRY_ATTEMPTS):
            recheck = await self._recheck(end - loop.time())
            if recheck is not None and (not strict or not recheck[0]):  # a start/end in flight: no hold to trust
                return recheck
            remaining = end - loop.time()
            if remaining <= 0:
                break

            async def call():
                return await asyncio.wait_for(fresh(), remaining)

            verdict, transport = await self._attempt(call, strict)
            if verdict is not None:
                return verdict
        recheck = await self._recheck(end - loop.time()) if loop.time() < end else None
        if recheck is not None:
            return recheck
        return False, UNREACHABLE_TEXT if transport else ESTOP_NOW

    async def stop_result(self) -> tuple[bool, str]:
        """(confirmed, text). confirmed False means the freeze was not confirmed; text says what the system could not do."""
        self._stop_epoch += 1
        self.pending = None
        ok, text = await self._do_stop(strict=self._inflight > 0)
        if ok:
            self._last_alarm = -1e9  # a confirmed hold ends the episode: the next failure must alarm again
        return ok, text

    async def stop(self) -> str:
        return (await self.stop_result())[1]

    def fast_stop(self, transcript: str, is_final: bool = True) -> asyncio.Task | None:
        """Voice fast path (skips the LLM): fires a stop task only on a FINAL transcript, when "freeze" or "halt" is one
        of its first two words and is not negated ("don't freeze"). Interim transcripts never fire.

        Debounced, and skipped while a fast stop is still in flight.
        """
        if not is_final:
            return None
        words = re.findall(r"[\w'-]+", normalize(transcript))[:2]
        fire = any(
            w in ("freeze", "halt") and not (i > 0 and words[i - 1] in NEGATIONS)
            for i, w in enumerate(words)
        )
        if not fire:
            return None
        if self._tasks:
            return None
        now = self._clock()
        if now - self._last_fast_stop < STOP_DEBOUNCE_S or now - self._fail_at < FAIL_REPEAT_S:
            return None
        self._last_fast_stop = now
        task = asyncio.create_task(self.stop_result())
        self._fast_turn = {"words": tokens(transcript), "at": now, "task": task, "claimed": None}
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        def _rearm(t) -> None:  # a failed fast stop must not block an immediate repeat
            if t.cancelled() or t.exception() is not None or not t.result()[0]:
                self._last_fast_stop = -1e9
                self._fail_at = self._clock()

        task.add_done_callback(_rearm)
        return task

    def fast_owned(self, text: str, turn_id: str | None = None) -> asyncio.Task | None:
        """The fast-path freeze task that already handled this user message, else None.

        A message is owned when its words equal the transcript the fast path fired on, within FAST_OWN_S, and no OTHER
        message has claimed that fire (a repeat the fast path debounced away is a new message and gets its own freeze).
        """
        f = self._fast_turn
        if f is None or self._clock() - f["at"] > FAST_OWN_S or tokens(text) != f["words"]:
            return None
        if f["claimed"] not in (None, turn_id):
            return None
        f["claimed"] = turn_id if turn_id is not None else f["claimed"]
        return f["task"]

    async def status(self) -> str:
        try:
            s = await self.client.status()
        except StationError as exc:
            return f"Could not read station status: {exc.message}"
        except Exception as exc:  # noqa: BLE001
            return f"Could not read station status: {exc!r}"
        if "running" in s:  # harness helper status
            keys = ("running", "state", "stage", "dry_run", "score", "alarm", "error", "age_s")
            vals = {k: (("hold NOT confirmed" if hold_unconfirmed(s) else "holding") if k == "state" and s.get(k) == "frozen" else s.get(k))
                    for k in keys}
            line = "Harness status: " + ", ".join(f"{k}={v}" for k, v in vals.items())
            if s.get("holders"):
                line += f". The arms are held by {', '.join(map(str, s['holders']))} (not ours)."
            elif s.get("untracked_pids"):
                line += ". Another harness process is running on the station (not ours)."
            return scrub_text(line)
        keys = ("state", "prompt", "checkpoint", "arms", "last_error", "run_elapsed_s")
        return "Station status: " + ", ".join(f"{k}={s.get(k)}" for k in keys)


def _data_failed(data) -> bool:
    """A helper result that says ok and it is not True is a failure; no ok key (station results) is success."""
    return isinstance(data, dict) and "ok" in data and data["ok"] is not True


def _error_text(data: dict) -> str:
    err = data.get("last_error") or data.get("error")
    if isinstance(err, dict):
        err = err.get("message") or str(err)
    return scrub_text(str(err)) if err else "unknown error"


def _summarize(action: PendingAction, res: StationResult, live: bool = True) -> str:
    if res.dry_run:
        return (
            f"DRY RUN: would {action.describe()}, but the station is not live so the arm did NOT move. "
            "Tell the human it was a dry run."
        )
    data = res.data if isinstance(res.data, dict) else {}
    state = data.get("state")
    if state == "error":
        return f"The station reports an ERROR: {_error_text(data)}. Tell the human."
    if action.kind == "end" and data.get("signalled") is False:
        if state == "done":
            return "The run had already finished; nothing moved."
        return "No run was active; nothing was sent."
    if not live and action.kind in ("relay", "end"):
        return (
            f"DRY RUN: the {'hill climb is starting' if action.kind == 'relay' else 'run is ending'} on the station with the "
            "arms read-only; they do not move. Tell the human it is a dry run."
        )
    if action.kind == "relay":
        return "Accepted: the hill climb is starting (not yet moving). Say it is starting."
    if action.kind == "end":
        if data.get("pending"):
            return "Sent: the run is ending; the arms will go home in a moment. Say so."
        return "Accepted: the run is ending and the arms are going home. Say so."
    if action.kind == "start":
        return f"Accepted: station state is {state} (starting up, not yet moving). Say it is starting."
    return f"Home move finished. Station state: {state}."
