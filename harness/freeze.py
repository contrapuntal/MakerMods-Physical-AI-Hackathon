"""Freeze control and the run's state.json (stdlib only, so it can be set up before the heavy imports).

SIGUSR1 freezes the arms in place: no home move, no switch-off. SIGINT/SIGTERM/SIGHUP end the run. The state file
(`<run_dir>/state.json`, with `~/.dreamlabs/current-harness-run` pointing at the run dir) lets other processes
(the voice agent) see where the run is.
"""

from __future__ import annotations

import json
import os
import logging
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

CURRENT_RUN_LINK = Path("~/.dreamlabs/current-harness-run").expanduser()


class FreezeRequested(Exception):
    """Raised inside the run when a freeze arrives."""


class _ReentrantEvent(threading.Event):
    """An Event on an RLock. Python signal handlers run on the main thread, possibly while it holds the event's
    lock (inside set/clear/wait); with the stock non-reentrant Lock the handler's set()/clear() would deadlock."""

    def __init__(self) -> None:
        super().__init__()
        self._cond = threading.Condition(threading.RLock())


class FreezeControl:
    def __init__(self) -> None:
        self.freeze: threading.Event = _ReentrantEvent()
        self.end_requested: threading.Event = _ReentrantEvent()
        self.on_freeze: Callable[[], None] | None = None

    def install_freeze_signal(self) -> None:
        signal.signal(signal.SIGUSR1, self._on_sigusr1)

    def _on_sigusr1(self, signum: int, frame: Any) -> None:
        self.end_requested.clear()
        self.freeze.set()
        if self.on_freeze is not None:
            try:
                self.on_freeze()
            except Exception:  # a handler must never raise into the code it interrupted
                log.exception("on_freeze callback failed")

    def request_end(self) -> None:
        self.end_requested.set()

    def begin_end(self) -> bool:
        """Clear the freeze before the end home move, so a new freeze can interrupt it.

        Returns False if a freeze arrived since the end was requested (it clears end_requested); the caller must
        then abort the end and go back to frozen.
        """
        self.freeze.clear()
        return self.end_requested.is_set()

    @property
    def frozen(self) -> bool:
        return self.freeze.is_set()

    def interruptible_sleep(self, dt: float) -> None:
        if self.freeze.wait(dt):
            raise FreezeRequested


class StateFile:
    def __init__(self, out_dir: Path, link: Path = CURRENT_RUN_LINK) -> None:
        self.out_dir = Path(out_dir)
        self.link = Path(link)
        self.data: dict[str, Any] = {}

    def update(self, **fields: Any) -> dict:
        payload = {**self.data, **fields, "updated_at": time.time()}  # committed only once the write succeeds
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "state.json"
        fd, tmp = tempfile.mkstemp(dir=self.out_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.data = payload
        return dict(self.data)

    def link_current(self) -> None:
        self.link.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.link.with_name(f"{self.link.name}.{os.getpid()}.tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(self.out_dir.resolve())
        os.replace(tmp, self.link)


def hold_measured(arms: dict, make_command: Callable[[Any, float], Any]) -> dict[str, Exception]:
    """Write each arm its own measured pose once, so it holds where it is.

    The gripper is clamped to [0, 1] (the arm refuses values outside). One arm failing does not stop the others;
    returns {name: error} for the arms that failed (empty when all held).
    """
    failed: dict[str, Exception] = {}
    for name, arm in arms.items():
        try:
            state = arm.read()
            arm.write(make_command(state.positions_rad, min(1.0, max(0.0, state.gripper))))
        except Exception as exc:
            log.error("hold %s failed: %s", name, exc)
            failed[name] = exc
    return failed


def first_alarm(arms: dict) -> str | None:
    for name, arm in arms.items():
        if not arm.is_alive():
            return f"{name}: not answering"
        alarm = arm.alarm()
        if alarm:
            return alarm if name in alarm else f"{name}: {alarm}"
    return None
