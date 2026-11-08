"""Freeze control and the run's state.json (stdlib only, so it can be set up before the heavy imports).

SIGUSR1 freezes the arms in place: no home move, no switch-off. SIGINT/SIGTERM/SIGHUP end the run. The state file
(`<run_dir>/state.json`, with `~/.dreamlabs/current-harness-run` pointing at the run dir) lets other processes
(the voice agent) see where the run is.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

CURRENT_RUN_LINK = Path("~/.dreamlabs/current-harness-run").expanduser()


class FreezeRequested(Exception):
    """Raised inside the run when a freeze arrives."""


class FreezeControl:
    def __init__(self) -> None:
        self.freeze = threading.Event()
        self.end_requested = threading.Event()
        self.on_freeze: Callable[[], None] | None = None

    def install_freeze_signal(self) -> None:
        signal.signal(signal.SIGUSR1, self._on_sigusr1)

    def _on_sigusr1(self, signum: int, frame: Any) -> None:
        self.end_requested.clear()
        self.freeze.set()
        if self.on_freeze is not None:
            self.on_freeze()

    def request_end(self) -> None:
        self.end_requested.set()

    def begin_end(self) -> None:
        """Clear the freeze before the end home move, so a new freeze can interrupt it."""
        self.freeze.clear()

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
        self.data.update(fields)
        self.data["updated_at"] = time.time()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "state.json"
        tmp = path.with_name(f"state.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, path)
        return dict(self.data)

    def link_current(self) -> None:
        self.link.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.link.with_name(f"{self.link.name}.{os.getpid()}.tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(self.out_dir.resolve())
        os.replace(tmp, self.link)


def hold_measured(arms: dict, make_command: Callable[[Any, float], Any]) -> None:
    """Write each arm its own measured pose once, so it holds where it is."""
    for arm in arms.values():
        state = arm.read()
        arm.write(make_command(state.positions_rad, state.gripper))


def first_alarm(arms: dict) -> str | None:
    for name, arm in arms.items():
        if not arm.is_alive():
            return f"{name}: not answering"
        alarm = arm.alarm()
        if alarm:
            return f"{name}: {alarm}"
    return None
