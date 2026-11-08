"""Remote helper for the harness voice control. STDLIB ONLY.

Runs ON the station as `python3 - <op> <args>` with this file on stdin (see harness_client.SshRunner).
Ops: status, start, end, freeze, check. Every op prints exactly one JSON object, even on errors.

Env HARNESS_REMOTE_HOME replaces `~` (tests only). In that mode only processes whose cmdline contains that
path are ever treated as runs or holders, so real processes are never matched or signalled.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HOLDER_PATTERNS = ("dhk-collect", "robots.station", "dreamcli robot", "robots.run", "check_arms.py")
START_SLACK_S = 5.0
CONTRACT_STRINGS = ("SIGUSR1 freezes the arms", "omit to bin every ball except")
FREEZE_POLL_S = 3.0  # inside harness_client.STOP_TOTAL_S (6 s); a slow LAN handshake must not turn a delivered signal into an alarm
HELP_TIMEOUT_S = 20
START_WAIT_S = 8.0
TMUX_SESSION = "match"
TERMINAL = ("frozen", "done", "error")
HEARTBEAT_STALE_S = 3.0  # `freezing` whose updated_at is this old is wedged, not slow (the harness beats every 1 s)


def _test_root() -> str | None:
    return os.environ.get("HARNESS_REMOTE_HOME") or None


def _home() -> Path:
    return Path(_test_root() or os.path.expanduser("~"))


def _expand(p: str) -> str:
    if p == "~" or p.startswith("~/"):
        return str(_home() / p[2:])
    return p


def _link() -> Path:
    return _home() / ".dreamlabs" / "current-harness-run"


def _lock_path() -> Path:
    return _home() / ".dreamlabs" / "arms.lock"


def _runs_dir() -> Path:
    return _home() / "hackathon_runs"


# ---------- /proc scanning ----------


def _tokens(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [t.decode("utf-8", "replace") for t in raw.split(b"\0") if t]


def _ppid(pid: int) -> int:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def _start_time(pid: int) -> float | None:
    """Process start as epoch seconds (stat field 22 / CLK_TCK + btime), or None."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        ticks = int(stat.rsplit(")", 1)[1].split()[19])
        btime = next(int(ln.split()[1]) for ln in Path("/proc/stat").read_text().splitlines() if ln.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _scoped(tokens: list[str]) -> bool:
    root = _test_root()
    return root is None or root in " ".join(tokens)


def _is_run(tokens: list[str]) -> bool:
    """`-m harness run ...` (adjacent tokens), or a `*harness.py run` script (the test fake). `--help` is not a run."""
    if "--help" in tokens:
        return False
    for i, t in enumerate(tokens):
        if t == "-m" and tokens[i + 1 : i + 3] == ["harness", "run"]:
            return True
        if t.endswith("harness.py") and tokens[i + 1 : i + 2] == ["run"]:
            return True
    return False


def _is_tmux(tokens: list[str]) -> bool:
    return bool(tokens) and os.path.basename(tokens[0]).startswith("tmux")


def _own_chain() -> set[int]:
    chain, pid = set(), os.getpid()
    while pid > 1 and pid not in chain:
        chain.add(pid)
        pid = _ppid(pid)
    return chain


def _verified(pid, started_at=None) -> bool:
    """pid alive, argv is a harness run, scoped to the test root, and not started after state.json's started_at
    (so a reused pid is rejected)."""
    if not isinstance(pid, int) or pid <= 1:
        return False
    tokens = _tokens(pid)
    if not tokens or not _is_run(tokens) or not _scoped(tokens):
        return False
    if isinstance(started_at, (int, float)):
        t0 = _start_time(pid)
        if t0 is not None and t0 > started_at + START_SLACK_S:
            return False
    return True


def _scan(exclude_pid: int | None) -> tuple[list[int], list[str]]:
    """(untracked live run pids, holder names), skipping own chain, tmux, and the tracked run and its descendants."""
    skip = _own_chain()
    procs: dict[int, list[str]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            toks = _tokens(int(entry))
            if toks:
                procs[int(entry)] = toks
    if exclude_pid:
        skip.add(exclude_pid)
        p, hops = _ppid(exclude_pid), 0  # wrappers (timeout, sudo, uv run ...) keep `-m harness run` in argv
        while p > 1 and hops < 20:
            skip.add(p)
            p, hops = _ppid(p), hops + 1
        for pid in procs:
            p, hops = _ppid(pid), 0
            while p > 1 and hops < 20:
                if p == exclude_pid:
                    skip.add(pid)
                    break
                p, hops = _ppid(p), hops + 1
    runs: list[int] = []
    found: list[str] = []
    for pid, toks in procs.items():
        if pid in skip or _is_tmux(toks) or not _scoped(toks):
            continue
        line = " ".join(toks)
        if _is_run(toks):
            runs.append(pid)
            if "harness run" not in found:
                found.append("harness run")
        elif any(a == "-m" and b == "harness" for a, b in zip(toks, toks[1:])) and "--help" not in toks:
            if "harness" not in found:
                found.append("harness")
        for name in HOLDER_PATTERNS:
            if name in line and name not in found:
                found.append(name)
    return sorted(runs), found


def _holders(exclude_pid: int | None = None) -> list[str]:
    return _scan(exclude_pid)[1]


def _untracked(exclude_pid: int | None) -> list[int]:
    return _scan(exclude_pid)[0]


_SYS_PIDFD_OPEN, _SYS_PIDFD_SEND_SIGNAL = 434, 424  # the same numbers on x86_64 and aarch64


def _syscall(number: int, *args: int) -> int:
    """A raw syscall for a Python built without os.pidfd_open / signal.pidfd_send_signal (stdlib ctypes only)."""
    import ctypes  # noqa: PLC0415

    libc = ctypes.CDLL(None, use_errno=True)
    res = libc.syscall(ctypes.c_long(number), *[ctypes.c_long(a) for a in args])
    if res < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return res


def _signal(pidfd: int, sig: int) -> None:
    """Signal through the pidfd ONLY. There is no numeric-pid fallback: a reused pid must never receive it."""
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, sig)
    else:
        res = _syscall(_SYS_PIDFD_SEND_SIGNAL, pidfd, sig, 0, 0)
        if res != 0:
            raise OSError("pidfd_send_signal failed")


def _open_pidfd(pid) -> tuple[int | None, str | None]:
    """(fd, None) or (None, reason). Never raises."""
    if not isinstance(pid, int) or pid <= 1:
        return None, "no valid pid"
    try:
        if hasattr(os, "pidfd_open"):
            return os.pidfd_open(pid), None
        return _syscall(_SYS_PIDFD_OPEN, pid, 0), None
    except (OSError, AttributeError, ValueError) as exc:
        return None, f"pidfd unavailable: {exc}"


def _strict_identity(pid, started_at) -> str | None:
    """None when pid is verified as the harness run that wrote `started_at` (start time included), else the reason.
    Unlike `_verified`, an unreadable or missing start time fails closed: signalling needs a positive identity."""
    if not isinstance(started_at, (int, float)):
        return "the state file has no started_at to verify the process against"
    if not _verified(pid, started_at):
        return "the pid is not a live harness run"
    t0 = _start_time(pid)
    if t0 is None:
        return "the process start time cannot be read"
    if t0 > started_at + START_SLACK_S:
        return "the pid was reused by a newer process"
    return None


def _acquire(pid, started_at) -> tuple[int | None, str | None]:
    """The pidfd of a positively verified run, else (None, reason). Opened BEFORE verifying, so the process verified is
    the process signalled."""
    fd, why = _open_pidfd(pid)
    if fd is None:
        return None, why
    bad = _strict_identity(pid, started_at)
    if bad:
        os.close(fd)
        return None, bad
    return fd, None


def _not_delivered(why: str, state) -> dict:
    return {"ok": False, "not_delivered": True, "signalled": False, "state": state,
            "error": f"not delivered: {why}; nothing was signalled"}


# ---------- run state ----------


def _current() -> dict | None:
    """The run the symlink points at: {"run_dir", "state": dict|None}. None when there is no symlink."""
    link = _link()
    try:
        run_dir = os.path.realpath(link) if link.is_symlink() else None
    except OSError:
        run_dir = None
    if not run_dir:
        return None
    try:
        state = json.loads((Path(run_dir) / "state.json").read_text())
        if not isinstance(state, dict):
            state = None
    except (OSError, ValueError):
        state = None
    return {"run_dir": run_dir, "state": state}


def _read_state(run_dir: str) -> dict | None:
    try:
        st = json.loads((Path(run_dir) / "state.json").read_text())
        return st if isinstance(st, dict) else None
    except (OSError, ValueError):
        return None


class _Pin:
    """The run a signal was sent to: its directory, pid and start time, fixed before signalling. Reads go to that
    directory only and count only while pid and started_at still match, so another run can never answer for it."""

    def __init__(self, run_dir: str | None, st: dict) -> None:
        self.run_dir = run_dir
        self.pid = st.get("pid")
        self.started_at = st.get("started_at")
        self.mismatch = False

    def read(self) -> dict:
        st = _read_state(self.run_dir) if self.run_dir else None
        if st is None:
            return {}
        if st.get("pid") != self.pid or st.get("started_at") != self.started_at:
            self.mismatch = True  # a different run wrote this directory's state: it acknowledges nothing
            return {}
        return st


LAUNCH_WINDOW_S = 45.0  # an empty run directory this young is a launch in progress; older is a failed one


def _launching(cur: dict | None) -> bool:
    run_dir = (cur or {}).get("run_dir")
    if not run_dir or (Path(run_dir) / "state.json").exists():
        return False
    try:
        return time.time() - os.stat(run_dir).st_mtime < LAUNCH_WINDOW_S
    except OSError:
        return False


def _hold_unconfirmed(st: dict) -> bool:
    if st.get("state") != "frozen":
        return False
    if st.get("hold_ok") is not None:
        return st["hold_ok"] is not True
    return bool(st.get("alarm") or st.get("fault"))


def _lock_free() -> bool:
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            fcntl.flock(f, fcntl.LOCK_UN)
            return True
    except OSError:
        return False


def _lock_holder_alive() -> bool:
    """Read-only lock check (never takes the lock): the file holds `<pid> <holder>`; busy when that pid is alive and
    has the lock file open (a stale pid left by a released lock, or a reused pid, reads as free)."""
    path = _lock_path()
    try:
        pid = int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    if pid <= 1 or not Path(f"/proc/{pid}").exists():
        return False
    try:
        target = os.path.realpath(path)
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == target:
                    return True
            except OSError:
                continue
        return False
    except OSError:
        return True  # alive but its fds are unreadable: assume it holds the lock


def op_status(_args: list[str]) -> dict:
    cur = _current()
    st = (cur or {}).get("state") or {}
    pid = st.get("pid")
    running = _verified(pid, st.get("started_at"))
    updated = st.get("updated_at")
    age = round(time.time() - updated, 1) if isinstance(updated, (int, float)) else None
    runs, holders = _scan(pid if running else None)
    freezing = st.get("state") == "freezing"
    since = st.get("freezing_since") if freezing else None
    return {
        "ok": True,
        "running": running,
        "pid": pid,
        "state": st.get("state"),
        "stage": st.get("stage"),
        "dry_run": st.get("dry_run"),
        "freeze_supported": st.get("freeze_supported"),
        "alarm": st.get("alarm"),
        "error": st.get("error"),
        "fault": st.get("fault"),
        "reason": st.get("reason"),
        "eight_ball_alert": st.get("eight_ball_alert"),
        "hold_ok": st.get("hold_ok"),
        "hold_failures": st.get("hold_failures"),
        "score": st.get("score"),
        "run_dir": (cur or {}).get("run_dir"),
        "age_s": age,
        # While `freezing` the harness refreshes updated_at every 1 s: its age is the heartbeat.
        "freezing_since": since,
        "heartbeat_age_s": age if freezing else None,
        "freezing_for_s": round(time.time() - since, 1) if isinstance(since, (int, float)) else None,
        "lock_free": not _lock_holder_alive(),
        "holders": holders,
        "untracked_pids": runs,
    }


def _verified_run() -> tuple[dict | None, str | None]:
    """(state, None) for a verified live run, else (state-or-None, reason)."""
    cur = _current()
    st = (cur or {}).get("state")
    if not st:
        return None, "no run found"
    if not _verified(st.get("pid"), st.get("started_at")):
        return st, "run pid is not alive or is not a harness run"
    return st, None


def _untracked_error(pids: list[int], holders: list[str] | None = None) -> dict:
    return {"ok": False, "error": f"untracked harness run pid {pids[0]}: end it with Ctrl-C in its window",
            "untracked_pids": pids, "holders": holders if holders is not None else _holders(), "not_ours": True}


def _wait_freeze(seconds: float, pin: _Pin, pre: str | None, signalled_at: float) -> tuple[str | None, bool]:
    """Poll the PINNED run after SIGUSR1. Returns (state, parking_seen). Stops at frozen/done/error or when parking shows.

    A `frozen` seen after a non-frozen start counts only if it was written after the signal (updated_at), so an old
    record cannot acknowledge this freeze; a state from a different run is never read (pin.mismatch)."""
    deadline = time.monotonic() + seconds
    while True:
        st = pin.read()
        state = st.get("state")
        if pin.mismatch:
            return None, False
        if state == "frozen" and pre != "frozen":
            upd = st.get("updated_at")
            if not isinstance(upd, (int, float)) or upd < signalled_at - 0.5:
                state = "stale_frozen"
        if state == "parking":
            return state, True
        if state in TERMINAL or time.monotonic() >= deadline:
            return state, False
        time.sleep(0.05)


PARK_ERROR = "the park cannot be stopped by voice; wait for it to finish"
ABORTED_NOTE = "aborted before the arms moved"
PRE_OPEN_REASONS = ("frozen_before_open", "ended_before_open")


def _aborted_before_open(st: dict) -> bool:
    """A `done` run that never opened an arm (the harness writes the reason; older copies only the error text)."""
    if st.get("state") != "done":
        return False
    if st.get("reason") is not None:
        return st["reason"] in PRE_OPEN_REASONS
    return "before the arms opened" in str(st.get("error") or "")


def op_freeze(_args: list[str]) -> dict:
    """Pid check and signal in one invocation, with no gap between them.

    Every verified, live, freeze-supporting run gets SIGUSR1 whatever its state (Ruling 24): the state file can lag
    an end request that is already being handled, so "frozen" on disk proves nothing. A repeat is harmless.
    """
    cur = _current()
    st = (cur or {}).get("state") or {}
    pid = st.get("pid")
    pin = _Pin((cur or {}).get("run_dir"), st)  # directory and identity fixed before anything is signalled
    fd, fd_why = _open_pidfd(pid)  # opened before verification so the verified process is the one signalled
    signalled_at = time.time()
    try:
        alive = _verified(pid, st.get("started_at"))
        other, held = _scan(pid if alive else None)
        if not st:
            if _launching(cur):
                # `start` has repointed the link at a new run directory whose process has not written state.json yet:
                # nothing can be signalled, but a run is about to exist, so "nothing is running" would be false.
                return {"ok": False, "launching": True, "state": None, "signalled": False,
                        "error": "a run is launching and has no state yet; the hold cannot be delivered until it does"}
            if other:
                return _untracked_error(other, held)
            if held or _lock_holder_alive():  # something else may be driving the arms: say what, never signal it
                return {"ok": False, "error": "no harness run found; another program holds the arms", "state": None,
                        "holders": held, "untracked_pids": [], "not_ours": True}
            return {"ok": True, "state": None, "signalled": False, "nothing_running": True,
                    "holders": [], "untracked_pids": []}
        if not alive:
            if other:
                return _untracked_error(other, held)
            state = st.get("state")
            if state == "frozen":
                return {"ok": False, "error": "run is marked frozen but its process is gone", "state": "frozen"}
            if held and state in ("done", "error", "frozen"):  # our run is over, but another program holds the arms
                return {"ok": False, "error": "our run is over; another program holds the arms", "state": state,
                        "holders": held, "untracked_pids": [], "not_ours": True}
            if state == "done":
                return {"ok": True, "state": "done", "signalled": False, "already_over": True}
            if state == "error":
                return {"ok": False, "error": f"the run ended in error: {st.get('error') or 'unknown'}", "state": "error"}
            return {"ok": False, "error": "run pid is not alive or is not a harness run", "state": state}
        if st.get("freeze_supported") is not True:
            if other:
                return _untracked_error(other, held)
            return {"ok": False, "error": "this harness copy does not support freeze", "state": st.get("state"), "final": True}
        # Fail closed: no pidfd, or no positive identity (incl. start time), means no signal at all.
        why = fd_why if fd is None else _strict_identity(pid, st.get("started_at"))
        if why:
            return _not_delivered(why, st.get("state"))
        try:  # SIGUSR1 never moves the arms, so send it first even if other runs exist
            _signal(fd, signal.SIGUSR1)
        except OSError as exc:
            return {"ok": False, "error": f"signal failed: {exc}", "state": st.get("state")}
    finally:
        if fd is not None:
            os.close(fd)
    pre = st.get("state")
    if pre == "done":  # exit lag: the process is still winding down, but the run is over
        state, parked = "done", False
    elif pre == "parking":
        state, parked = pre, True
    else:
        state, parked = _wait_freeze(FREEZE_POLL_S, pin, pre, signalled_at)
    if pin.mismatch:
        return {"ok": False, "error": "the run's state changed to a different run while waiting; the hold is not confirmed",
                "state": None, "signalled": True}
    if other:
        return {**_untracked_error(other, held), "state": state, "signalled": True}
    done_now = pin.read()
    if pin.mismatch:
        return {"ok": False, "error": "the run's state changed to a different run while waiting; the hold is not confirmed",
                "state": None, "signalled": True}
    if (pre == "done" or state == "done") and _aborted_before_open(done_now):
        # Aborted before any arm opened (frozen or ended during start-up), whichever state the freeze saw first.
        return {"ok": True, "state": "done", "signalled": True, "note": ABORTED_NOTE}
    if pre == "done":
        return {"ok": True, "state": "done", "signalled": True, "already_over": True}
    if parked or state == "done":
        return {"ok": False, "error": PARK_ERROR, "state": state, "signalled": True, "final": True}
    if state == "frozen":
        if _hold_unconfirmed(done_now):
            failures = done_now.get("hold_failures")
            return {"ok": False, "hold_failed": True, "hold_ok": False, "state": "frozen", "signalled": True,
                    "hold_failures": failures if isinstance(failures, list) else None, "final": True,
                    "error": "frozen, but an arm did not take the hold: "
                             + str(done_now.get("alarm") or done_now.get("fault") or failures or "unknown")}
        return {"ok": True, "state": "frozen", "signalled": True}
    if state == "freezing":  # delivered and being handled; pending only while the harness's heartbeat is live
        updated = done_now.get("updated_at")
        beat = time.time() - updated if isinstance(updated, (int, float)) else None
        if beat is not None and beat >= HEARTBEAT_STALE_S:
            return {"ok": False, "error": f"freezing, but its heartbeat is {beat:.0f} s old; the hold is not confirmed",
                    "state": "freezing", "signalled": True}
        return {"ok": True, "state": "freezing", "signalled": True, "pending": True}
    if state == "starting":
        return {"ok": True, "state": "starting", "signalled": True,
                "note": "freeze accepted before the run started; it aborts before the arms move"}
    if state == "error":
        err = done_now.get("error") or st.get("error") or "unknown"
        return {"ok": False, "error": f"the run is in error: {err}; the hold is not confirmed", "state": "error",
                "signalled": True}
    return {"ok": False, "error": f"no frozen state within {FREEZE_POLL_S:g} s", "state": state, "signalled": True}


END_POLL_S = 2.5  # inside harness_client.END_TIMEOUT_S (6 s) with ssh latency to spare
END_BRIEF_POLL_S = 1.0
END_TAKEN_STATES = {"homing", "parking", "done", "error"}


def op_end(args: list[str]) -> dict:
    p = _Parser(prog="end", add_help=False)
    p.add_argument("--require-dry", action="store_true")
    a = p.parse_args(args)
    cur = _current()
    st = (cur or {}).get("state") or {}
    pid = st.get("pid")
    pin = _Pin((cur or {}).get("run_dir"), st)
    fd, fd_why = _open_pidfd(pid)
    try:
        if _verified(pid, st.get("started_at")):
            if a.require_dry and st.get("dry_run") is not True:
                return {"ok": False, "error": "refusing: the run is not a dry run and this agent is not live",
                        "state": st.get("state")}
            why = fd_why if fd is None else _strict_identity(pid, st.get("started_at"))
            if why:  # fail closed: the end moves the arms, so it needs a positive identity
                return _not_delivered(why, st.get("state"))
            try:
                _signal(fd, signal.SIGINT)
            except OSError as exc:
                return {"ok": False, "error": f"signal failed: {exc}", "state": st.get("state")}
        else:
            other = _untracked(None)  # nothing verified was signalled: an untracked run must not read as handled
            if other:
                return _untracked_error(other)
            if st.get("state") == "done":
                return {"ok": True, "state": "done", "signalled": False}
            return {"ok": False, "error": "run pid is not alive or is not a harness run" if st else "no run found",
                    "state": st.get("state")}
    finally:
        if fd is not None:
            os.close(fd)
    # Never re-send SIGINT here: a freeze said in between must win.
    was_frozen = st.get("state") == "frozen"
    # Frozen: _frozen_wait sees an end within about 0.6 s, so still-frozen after END_POLL_S means it was dropped.
    # Otherwise the end takes effect when the loop exits (a live ControlLoop teardown can take ~4 s): poll briefly
    # only to report homing if it is already there; a slow end is not a refusal.
    deadline = time.monotonic() + (END_POLL_S if was_frozen else END_BRIEF_POLL_S)
    seen = st.get("state")
    while True:
        time.sleep(0.1)
        now_st = pin.read()  # the pinned run only
        if pin.mismatch:
            return {"ok": False, "error": "the run's state changed to a different run while waiting; the end is unconfirmed",
                    "state": None, "signalled": True}
        seen = now_st.get("state", seen)
        if seen in END_TAKEN_STATES:
            return {"ok": True, "state": seen, "signalled": True}
        if time.monotonic() >= deadline:
            break
    if was_frozen:
        err = "the end was not taken: the arms are still holding. Say end the run again."
        return {"ok": False, "error": err, "state": seen, "signalled": True}
    return {"ok": True, "state": seen, "signalled": True, "pending": True,
            "note": "the end was sent; the run homes when the control loop shuts down (1-4 s)"}


# ---------- start ----------


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # never sys.exit: the op must still print JSON
        raise ValueError(message)


_STAGES = ("unplug", "stack", "balls")
_FLOAT_RE = re.compile(r"\d{1,2}(?:\.\d{1,4})?")
_VALUE_FLAGS = {  # flag -> validator of its one value
    "--color": lambda v: re.fullmatch(r"[a-z]+", v) is not None,
    **{f"--lower-{st}-cm": (lambda v: _FLOAT_RE.fullmatch(v) is not None and 0 <= float(v) <= 20) for st in _STAGES},
    **{f"--model-{st}": (lambda v: re.fullmatch(r"[A-Za-z0-9:._/-]+", v) is not None) for st in _STAGES},
    "--augment": lambda v: re.fullmatch(r"[a-z,]+", v) is not None,
    "--stack-pass": lambda v: v in ("6in", "3-cubes"),
    "--gravity-scale": lambda v: _FLOAT_RE.fullmatch(v) is not None and 0 <= float(v) <= 1.5,
    **{f"--gravity-{st}": (lambda v: _FLOAT_RE.fullmatch(v) is not None and 0 <= float(v) <= 1.5) for st in _STAGES},
    "--pull-cables-until": lambda v: _FLOAT_RE.fullmatch(v) is not None and 0 <= float(v) <= 300,
    "--detector": lambda v: v in ("hsv", "yolo", "both"),
    "--start-stage": lambda v: v in _STAGES,  # practice only: a real match starts at unplug
}
_BOOL_FLAGS = ("--rod-on-tower", "--all-colors")


def validate_extra_args(text: str | None) -> tuple[list[str], str | None]:
    """(tokens, None) when every token of the string is on the allowlist, else ([], reason). Anything else, including
    --checkpoint, --dry-run, --time-limit, --calibration, --out and the --*-text flags, refuses the whole string."""
    if not text or not text.strip():
        return [], None
    try:
        toks = shlex.split(text)
    except ValueError as exc:
        return [], f"bad --extra-args: {exc}"
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _BOOL_FLAGS:
            out.append(t)
            i += 1
        elif t in _VALUE_FLAGS:
            if i + 1 >= len(toks) or not _VALUE_FLAGS[t](toks[i + 1]):
                return [], f"bad --extra-args: {t} needs a valid value"
            out += [t, toks[i + 1]]
            i += 2
        else:
            return [], f"bad --extra-args: {t!r} is not an allowed run flag"
    return out, None


def _start_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="start", add_help=False)
    p.add_argument("--dir", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--calibration", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--time-limit", type=float)
    p.add_argument("--extra-args", default="")
    return p


def _new_run_dir() -> Path:
    base = _runs_dir()
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i in range(100):
        path = base / (stamp if i == 0 else f"{stamp}-{i}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not create a run directory")


def _point_link(run_dir: Path) -> None:
    """Atomically re-point ~/.dreamlabs/current-harness-run at the new (still empty) run directory."""
    link = _link()
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(f"{link.name}.start{os.getpid()}")
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    os.symlink(run_dir, tmp)
    os.replace(tmp, link)


def _undo_link(previous: str | None) -> None:
    """Put the link back to `previous` (or remove it when there was none). Best effort."""
    try:
        if previous is None:
            os.unlink(_link())
        else:
            tmp = _link().with_name(_link().name + f".undo{os.getpid()}")
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            os.symlink(previous, tmp)
            os.replace(tmp, _link())
    except OSError:
        pass


def _run_ready(run_dir: Path) -> bool:
    """state.json exists AND the link resolves to this run directory."""
    return (run_dir / "state.json").exists() and os.path.realpath(_link()) == os.path.realpath(run_dir)


def _launch_inner(workdir: str, python: str, checkpoint: str | None, calibration: str, dry_run: bool,
                  run_dir: str, time_limit: float | None = None, extra: list[str] | None = None) -> str:
    """The shell line the tmux/setsid launch runs. Pure: paths are already expanded."""
    q = shlex.quote
    parts = [q(python), "-m", "harness", "run"]
    if checkpoint:
        parts += ["--checkpoint", q(checkpoint)]
    parts += ["--calibration", q(calibration)]
    parts += [q(x) for x in extra or []]
    if dry_run:
        parts.append("--dry-run")
    if time_limit is not None:
        parts += ["--time-limit", f"{time_limit:g}"]
    parts += ["--out", q(run_dir)]
    return f"cd {q(workdir)} && exec {' '.join(parts)} > {q(run_dir + '/run.log')} 2>&1"


def _tail(path: Path, n: int = 400) -> str:
    try:
        return path.read_text(errors="replace")[-n:].strip()
    except OSError:
        return ""


def _help_missing(workdir: str, python: str) -> tuple[list[str], str | None]:
    """(contract strings missing from `harness run --help`, error). Help is whitespace-normalised at COLUMNS=1000."""
    try:
        res = subprocess.run(
            [python, "-m", "harness", "run", "--help"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=HELP_TIMEOUT_S,
            env={**os.environ, "COLUMNS": "1000"},
        )
        help_text = " ".join((res.stdout + res.stderr).split())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"could not run `harness run --help`: {exc}"
    return [s for s in CONTRACT_STRINGS if s not in help_text], None


def _tmux_available() -> bool:
    """Test mode (HARNESS_REMOTE_HOME) and the fake launcher never touch tmux, so a test cannot reach a real session."""
    return not _test_root() and not os.environ.get("HARNESS_FAKE_CMD") and bool(shutil.which("tmux"))


def _tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=10)


def _match_session_state() -> str:
    """"none" (no tmux or no `match` session), "dead" (every pane dead), or "live" (anything else, including doubt)."""
    if not _tmux_available():
        return "none"
    try:
        if _tmux("has-session", "-t", "=" + TMUX_SESSION).returncode != 0:
            return "none"
        r = _tmux("list-panes", "-s", "-t", "=" + TMUX_SESSION, "-F", "#{pane_dead}")
    except (OSError, subprocess.TimeoutExpired):
        return "live"
    panes = r.stdout.split() if r.returncode == 0 else []
    return "dead" if panes and all(x == "1" for x in panes) else "live"


def op_check(args: list[str]) -> dict:
    """Read-only preflight: help contract plus status. Never signals or starts anything."""
    p = _Parser(prog="check", add_help=False)
    p.add_argument("--dir", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--calibration")
    p.add_argument("--extra-args", default="")
    a = p.parse_args(args)
    _, bad_extra = validate_extra_args(a.extra_args)
    missing, err = _help_missing(_expand(a.dir), _expand(a.python))
    st = op_status([])
    return {
        "ok": True,
        "extra_args_ok": bad_extra is None,
        "extra_args_error": bad_extra,
        "calibration_ok": None if not a.calibration else os.path.exists(_expand(a.calibration)),
        "contract_ok": err is None and not missing,
        "missing": missing,
        "help_error": err,
        "running": st["running"],
        "lock_free": _lock_free(),
        "holders": st["holders"],
        "untracked_pids": st["untracked_pids"],
        "match_session": _match_session_state(),
    }


def op_start(args: list[str]) -> dict:
    a = _start_parser().parse_args(args)
    if a.time_limit is not None and not (0 < a.time_limit < 1e7):
        return {"ok": False, "error": f"invalid --time-limit {a.time_limit!r}"}
    workdir, python, calibration = _expand(a.dir), _expand(a.python), _expand(a.calibration)
    checkpoint = _expand(a.checkpoint) if a.checkpoint else None
    extra, bad = validate_extra_args(a.extra_args)  # no checkpoint is fine: the harness uses its per-stage defaults
    if bad:
        return {"ok": False, "error": bad}
    st, why = _verified_run()
    if not why:
        return {"ok": False, "error": "a harness run is already active", "state": st.get("state")}
    held = _holders()
    if held:
        return {"ok": False, "error": "arms held by: " + ", ".join(held), "holders": held}
    if not _lock_free():
        return {"ok": False, "error": "arms lock is busy"}
    use_tmux = _tmux_available()
    match = _match_session_state()
    if match == "live":
        return {"ok": False, "error": f"tmux session '{TMUX_SESSION}' has a live pane; end or kill it by hand"}
    if match == "dead":  # nothing runs in it, nothing holds the arms (checked above): clear it so tmux can reuse the name
        if _match_session_state() != "dead":  # re-check right before killing: never kill a session that came alive
            return {"ok": False, "error": f"tmux session '{TMUX_SESSION}' changed state; try again"}
        if _tmux("kill-session", "-t", "=" + TMUX_SESSION).returncode != 0:
            return {"ok": False, "error": f"could not clear the dead tmux session '{TMUX_SESSION}'; kill it by hand"}
    run_dir = _new_run_dir()
    log = run_dir / "run.log"
    try:
        previous = os.readlink(_link())
    except OSError:
        previous = None
    _point_link(run_dir)  # BEFORE the slow help check and the spawn: a freeze must not read the previous run's `done`

    def fail(error: str) -> dict:  # nothing is running: put the previous run's link back (M1), on every failed path
        _undo_link(previous)
        return {"ok": False, "error": error, "run_dir": str(run_dir)}

    try:
        missing, err = _help_missing(workdir, python)
        if err:
            return fail(err)
        if missing:
            return fail("harness copy lacks freeze support (help missing: %s)" % "; ".join(missing))
        fake = os.environ.get("HARNESS_FAKE_CMD")
        if fake:
            argv = (shlex.split(fake) + ["run"] + extra + (["--dry-run"] if a.dry_run else [])
                    + (["--time-limit", f"{a.time_limit:g}"] if a.time_limit is not None else [])
                    + ["--out", str(run_dir)])
            with open(log, "ab") as lf:
                subprocess.Popen(
                    argv, cwd=workdir, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        else:
            inner = _launch_inner(workdir, python, checkpoint, calibration, a.dry_run, str(run_dir), a.time_limit, extra)
            if use_tmux:
                r = subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION, inner],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    return fail(f"tmux failed: {r.stderr.strip()}")
            else:
                subprocess.Popen(
                    ["setsid", "nohup", "sh", "-c", inner],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
    except (OSError, subprocess.SubprocessError) as exc:
        return fail(f"could not launch the run: {type(exc).__name__}: {exc}")
    # The harness installs its handler and writes state.json before the heavy imports.
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline and not _run_ready(run_dir):
        time.sleep(0.05)
    if not (run_dir / "state.json").exists():
        return fail("run did not write state.json: " + (_tail(log) or "empty run.log"))
    if not _run_ready(run_dir):
        return {"ok": False, "error": "the current-harness-run link no longer points at the new run", "run_dir": str(run_dir)}
    return {"ok": True, "run_dir": str(run_dir)}


OPS = {"status": op_status, "start": op_start, "end": op_end, "freeze": op_freeze, "check": op_check}


def main(argv: list[str]) -> int:
    try:
        if not argv or argv[0] not in OPS:
            out = {"ok": False, "error": f"unknown op: {argv[:1]}"}
        else:
            out = OPS[argv[0]](argv[1:])
    except BaseException as exc:  # noqa: BLE001 - always print JSON
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
