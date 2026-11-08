"""Stand-in for `harness run`, honouring the shared contract (state.json, symlink, signals).

argv: run [--dry-run] --out D. SIGUSR1 -> frozen, SIGINT/SIGTERM -> homing, parking, done.
Every received signal is appended to <D>/signals.txt. Contract: `starting` = no arm touched; a live run writes `homing`
before "opening the arms" and `running` after the start-up home; a dry run writes `running` straight away.
Env: FAKE_FREEZE_DELAY delays the frozen write; FAKE_STAY_STARTING stays in `starting`; FAKE_START_DELAY sleeps before
the first state write; FAKE_HOME_DELAY lengthens `homing`; FAKE_END_DELAY is the slow-end window (a SIGUSR1 inside it
re-holds the run and drops the pending end); FAKE_DROP_END_WHILE_FROZEN ignores SIGINT while frozen (the end is
dropped, as when it lands before the hold); FAKE_ERROR_STATE writes `error` and stays alive; FAKE_WRITE_FREEZING
writes `freezing` immediately on SIGUSR1 and `frozen` after FAKE_FREEZE_DELAY (the real harness's `freezing` state).
`--time-limit N` is recorded in state.json as `time_limit`.
Home for the symlink is HARNESS_REMOTE_HOME (else ~). Test-only: nothing here touches hardware.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    assert argv and argv[0] == "run"
    dry = "--dry-run" in argv
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    freeze_delay = float(os.environ.get("FAKE_FREEZE_DELAY", "0"))
    end_delay = float(os.environ.get("FAKE_END_DELAY", "0"))
    home_delay = float(os.environ.get("FAKE_HOME_DELAY", "0"))
    limit = float(argv[argv.index("--time-limit") + 1]) if "--time-limit" in argv else None
    started = time.time()
    st = {
        "version": 1, "state": "starting", "pid": os.getpid(), "started_at": started, "stage": None,
        "dry_run": dry, "freeze_supported": True, "score": None, "alarm": None, "error": None,
        "updated_at": started,
        "time_limit": limit,
    }

    def write(**kw):
        st.update(kw, updated_at=time.time())
        tmp = out / "state.json.tmp"
        tmp.write_text(json.dumps(st))
        os.replace(tmp, out / "state.json")

    def note(name):
        with open(out / "signals.txt", "a") as f:
            f.write(name + "\n")

    events = []
    signal.signal(signal.SIGUSR1, lambda *_: (note("SIGUSR1"), events.append("freeze")))
    signal.signal(signal.SIGINT, lambda *_: (note("SIGINT"), events.append("end")))
    signal.signal(signal.SIGTERM, lambda *_: (note("SIGTERM"), events.append("end")))

    time.sleep(float(os.environ.get("FAKE_START_DELAY", "0")))
    write()
    home = Path(os.environ.get("HARNESS_REMOTE_HOME") or os.path.expanduser("~"))
    link = home / ".dreamlabs" / "current-harness-run"
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link.with_name(link.name + f".{os.getpid()}")
    os.symlink(out, tmp_link)
    os.replace(tmp_link, link)
    if os.environ.get("FAKE_STAY_STARTING"):  # stays in `starting`; a freeze/end aborts before the arms open
        while True:
            time.sleep(0.05)
            if events:
                write(state="done", error="frozen before the arms opened" if events[0] == "freeze"
                      else "ended before the arms opened",
                      reason="frozen_before_open" if events[0] == "freeze" else "ended_before_open")
                return 0
    held = False
    if not dry:
        write(state="homing")
        end = time.time() + home_delay
        while time.time() < end:
            time.sleep(0.02)
            if "freeze" in events:  # a freeze during the start-up home: hold as soon as it can
                events.remove("freeze")
                held = True
                break
    write(state="frozen" if held else "running", **({} if held else {"stage": "unplug"}))
    if os.environ.get("FAKE_ERROR_STATE"):  # an error with the process still alive (e.g. holding while it unwinds)
        write(state="error", error="boom")
        while True:
            time.sleep(0.05)

    while True:
        time.sleep(0.02)
        if not events:
            continue
        ev = events.pop(0)
        if ev == "freeze":
            if st["state"] in ("running", "frozen"):
                if os.environ.get("FAKE_WRITE_FREEZING"):
                    write(state="freezing", freezing_since=time.time())
                    stop_at = time.time() + freeze_delay
                    while time.time() < stop_at:  # the real harness's watcher thread: heartbeat every 0.5 s here
                        time.sleep(min(0.5, max(0.0, stop_at - time.time())))
                        if time.time() < stop_at:
                            write()
                else:
                    time.sleep(freeze_delay)
                write(state="frozen")
        else:
            if st["state"] == "frozen" and os.environ.get("FAKE_DROP_END_WHILE_FROZEN"):
                continue  # the end is dropped; the run keeps holding
            if end_delay:
                deadline, rehold = time.time() + end_delay, False
                while time.time() < deadline:
                    time.sleep(0.02)
                    if "freeze" in events:
                        events.remove("freeze")
                        rehold = True
                if rehold:  # SIGUSR1 inside the slow-end window: stay held, the end is dropped
                    write(state="frozen")
                    continue
            for s in ("homing", "parking"):
                write(state=s)
                time.sleep(0.2)
            write(state="done", score=0)
            return 0


if __name__ == "__main__":
    sys.exit(main())
