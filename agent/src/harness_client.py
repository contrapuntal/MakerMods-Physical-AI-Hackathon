"""Client for the harness voice control: drives src/harness_remote.py on the station.

SshRunner: one ssh per call over its own ControlMaster (not ops/tunnels.sh's), helper source on stdin. LocalRunner: same, locally (tests, fake mode).
Runners never raise: a timeout or failure comes back as {"ok": False, "error": ...}.
HarnessClient.stop() IS the freeze: always sent, bounded to STOP_TOTAL_S, never raises.
Not live: start launches `harness run --dry-run` (arms read-only) and end is sent with --require-dry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from station_client import StationResult

logger = logging.getLogger("harness_client")

STOP_TOTAL_S = 6.0  # helper FREEZE_POLL_S (3 s) plus a slow ssh handshake; measure on the station
STATUS_TIMEOUT_S = 5.0
START_TIMEOUT_S = 30.0
END_TIMEOUT_S = 6.0
HELPER_PATH = Path(__file__).with_name("harness_remote.py")


DEFAULT_TIME_LIMIT_S = 3600.0  # the run does not end by itself: the referee scores, then the operator ends it


def _env_time_limit() -> float:
    try:
        v = float(os.environ.get("HARNESS_TIME_LIMIT", DEFAULT_TIME_LIMIT_S))
    except ValueError:
        return DEFAULT_TIME_LIMIT_S
    return v if 0 < v < 1e7 else DEFAULT_TIME_LIMIT_S


def _fail(msg: str) -> dict[str, Any]:
    """A transport failure (ssh, timeout, bad output): the helper never gave a verdict."""
    return {"ok": False, "error": msg, "transport": True}


async def _exec(spawn, argv: list[str], env: dict | None, timeout_s: float) -> dict[str, Any]:
    try:
        source = HELPER_PATH.read_bytes()
        proc = await spawn(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"could not start helper: {exc}")
    try:
        out, err = await asyncio.wait_for(proc.communicate(source), timeout_s)
    except (TimeoutError, asyncio.CancelledError) as exc:
        try:
            proc.kill()
            await asyncio.wait_for(proc.wait(), 1.0)
        except Exception:  # noqa: BLE001
            pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        return _fail(f"helper did not answer within {timeout_s:g} s")
    except Exception as exc:  # noqa: BLE001
        return _fail(f"helper failed: {exc}")
    lines = [ln for ln in out.decode("utf-8", "replace").splitlines() if ln.strip()]
    try:
        data = json.loads(lines[-1]) if lines else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        tail = err.decode("utf-8", "replace").strip()[-200:]
        return _fail(f"bad helper output (rc={proc.returncode}): {tail or 'no output'}")
    return data


class SshRunner:
    """ssh <opts> -- <host> python3 - <op> <args>, over a dedicated ControlMaster (a dead master falls back to a new connection)."""

    def __init__(self, host: str, spawn=asyncio.create_subprocess_exec, fresh: bool = False) -> None:
        self.host = host
        self._spawn = spawn
        self.fresh = fresh

    def fresh_runner(self) -> "SshRunner":
        """Same host, but a brand-new connection (ControlPath=none): a half-dead master cannot block it."""
        return SshRunner(self.host, self._spawn, fresh=True)

    def argv(self, op: str, args: list[str]) -> list[str]:
        remote = " ".join(["python3", "-", op, *[shlex.quote(a) for a in args]])
        sock = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "pah2026-helper-%h.sock")
        if self.fresh:
            opts = ["BatchMode=yes", "ConnectTimeout=2", "ControlMaster=no", "ControlPath=none",
                    "ServerAliveInterval=2", "ServerAliveCountMax=2"]
        else:
            opts = ["BatchMode=yes", "ConnectTimeout=2", "ControlMaster=auto", f"ControlPath={sock}",
                    "ControlPersist=10m", "ServerAliveInterval=2", "ServerAliveCountMax=2"]
        return ["ssh", *[x for o in opts for x in ("-o", o)], "--", self.host, remote]

    async def call(self, op: str, args: list[str], timeout_s: float) -> dict[str, Any]:
        return await _exec(self._spawn, self.argv(op, args), None, timeout_s)


class LocalRunner:
    """Runs the helper locally with `env` merged over os.environ."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self.env = env or {}

    def fresh_runner(self) -> "LocalRunner":
        return self  # no connection to go stale

    async def call(self, op: str, args: list[str], timeout_s: float) -> dict[str, Any]:
        argv = [sys.executable, "-", op, *args]
        return await _exec(asyncio.create_subprocess_exec, argv, {**os.environ, **self.env}, timeout_s)


class HarnessClient:
    def __init__(
        self,
        runner,
        *,
        live: bool,
        harness_dir: str,
        calibration: str,
        checkpoint: str | None,
        python: str = "~/dream-env/bin/python",
        time_limit_s: float | None = None,
        run_args: str | None = None,
    ) -> None:
        # Extra `harness run` flags (HARNESS_RUN_ARGS); the helper validates them against an allowlist.
        self.run_args = (os.environ.get("HARNESS_RUN_ARGS", "") if run_args is None else run_args).strip()
        self.runner = runner
        self._live = live
        self.harness_dir = harness_dir
        self.calibration = calibration
        self.checkpoint = checkpoint
        self.python = python
        self.time_limit_s = time_limit_s if time_limit_s is not None else _env_time_limit()
        self._freeze_task: asyncio.Task | None = None

    @property
    def live(self) -> bool:
        return self._live

    async def _safe_call(self, op: str, args: list[str], timeout_s: float, runner=None) -> dict[str, Any]:
        try:
            data = await asyncio.wait_for((runner or self.runner).call(op, args, timeout_s), timeout_s)
            return data if isinstance(data, dict) else _fail("unexpected helper reply")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return _fail(f"{op} did not answer within {timeout_s:g} s")
        except Exception as exc:  # noqa: BLE001
            return _fail(f"{type(exc).__name__}: {exc}")

    async def status(self, fresh: bool = False) -> dict[str, Any]:
        runner = self.runner.fresh_runner() if fresh and hasattr(self.runner, "fresh_runner") else None
        data = await self._safe_call("status", [], STATUS_TIMEOUT_S, runner)
        if not data.get("ok", True):
            return {"ok": False, "running": None, "error": data.get("error", "status failed")}
        return data

    async def status_fresh(self) -> dict[str, Any]:
        return await self.status(fresh=True)

    def _start_args(self, dry_run: bool = False) -> list[str]:
        args = ["--dir", self.harness_dir, "--python", self.python, "--calibration", self.calibration]
        args += ["--time-limit", f"{self.time_limit_s:g}"]
        if self.run_args:
            args.append(f"--extra-args={self.run_args}")
        if self.checkpoint:
            args += ["--checkpoint", self.checkpoint]
        if dry_run:
            args.append("--dry-run")
        return args

    async def start_relay(self, dry_run: bool = False) -> StationResult:
        """Live: launch the run. Not live: the run is launched with --dry-run FORCED (arms read-only), never a no-op."""
        dry_run = dry_run or not self.live
        args = self._start_args(dry_run)
        sent = {"op": "start", "args": args}
        return StationResult(False, "SSH", "start", sent, await self._safe_call("start", args, START_TIMEOUT_S))

    async def end(self) -> StationResult:
        """Not live: the helper itself refuses unless the run's state.json says dry_run (--require-dry)."""
        args = [] if self.live else ["--require-dry"]
        sent = {"op": "end", "args": args}
        return StationResult(False, "SSH", "end", sent, await self._safe_call("end", args, END_TIMEOUT_S))

    async def _freeze_once(self, timeout_s: float = STOP_TOTAL_S, runner=None) -> StationResult:
        sent = {"op": "freeze", "args": []}
        try:
            data = await asyncio.wait_for(self._safe_call("freeze", [], timeout_s, runner), timeout_s)
        except TimeoutError:
            data = _fail(f"freeze did not complete within {timeout_s:g} s")
        return StationResult(False, "SSH", "freeze", sent, data)

    async def stop_fresh(self, timeout_s: float = STOP_TOTAL_S) -> StationResult:
        """A retry of the freeze over a fresh connection (not the persistent master). Never raises."""
        runner = self.runner.fresh_runner() if hasattr(self.runner, "fresh_runner") else None
        try:
            return await self._freeze_once(timeout_s, runner)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return StationResult(False, "SSH", "freeze", None, _fail(f"{type(exc).__name__}: {exc}"))

    async def stop(self) -> StationResult:
        """The freeze. Always sent, bounded to STOP_TOTAL_S, never raises.

        Concurrent callers share one in-flight freeze (one SIGUSR1); a finished or timed-out one never blocks the next.
        """
        task = self._freeze_task
        if task is None or task.done():
            task = self._freeze_task = asyncio.ensure_future(self._freeze_once())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return StationResult(False, "SSH", "freeze", None, _fail(f"{type(exc).__name__}: {exc}"))
