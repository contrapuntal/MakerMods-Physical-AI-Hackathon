"""One Dream-1 control loop on the real station, held open for a whole match, with a thread-safe handle.

Opens everything the way `dreamcli robot run` does (`robots.run.main`): cameras, the Dream client, the arms last (read
only in a dry run), home, then Dream's own `ControlLoop`. The loop runs until `stop` is set; then the arms hold, move
home and switch off. Other threads (the station server's HTTP handlers) talk to it through `StationHandle`:
`set_task()` is marshalled onto the loop's event loop, as `ControlLoop.set_task` requires, and `snapshot()` reads the
newest overhead frame and arm state.

Only this module and dream_executor.py import Dream's `robots` package.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import pathlib
import threading
import time
from collections.abc import Awaitable, Callable

import numpy as np

_log = logging.getLogger(__name__)

STATION_CONFIG = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
OVERHEAD = "scene"  # the overhead camera's name in the station config
HOLD_STILL = "hold the arms still"


@dataclasses.dataclass
class Snapshot:
    rgb: np.ndarray  # the overhead frame, RGB
    timestamp_s: float
    grip: dict[str, float]  # each arm's gripper opening, 0 closed to 1 open


class StationHandle:
    """What the server needs from a running station. FakeStation in the tests implements the same three methods."""

    def __init__(self) -> None:
        self._aloop: asyncio.AbstractEventLoop | None = None
        self._control = None
        self.ready = threading.Event()
        self.fault: BaseException | None = None

    def set_task(self, text: str) -> None:
        if self._aloop is None or self._control is None:
            raise RuntimeError("the station is not running")
        self._aloop.call_soon_threadsafe(self._control.set_task, text)

    def task(self) -> str:
        return "" if self._control is None else self._control.task

    def snapshot(self) -> Snapshot | None:
        loop = self._control
        if loop is None:
            return None
        frame, state = loop.video.get(OVERHEAD), loop.state
        if frame is None or state is None:
            return None
        return Snapshot(frame.data, frame.timestamp_s,
                        {"left": float(state.left.gripper), "right": float(state.right.gripper)})


async def run_station(handle: StationHandle, checkpoint: str, *, dry_run: bool, stop: asyncio.Event,
                      station_config: pathlib.Path = STATION_CONFIG,
                      on_ready: Callable[[], Awaitable[None]] | None = None) -> None:
    """Opens the station, runs the loop on `HOLD_STILL` until `stop`, then homes and switches the arms off."""
    from robots import bodies, control, followers, home
    from robots import config as rconfig
    from robots import run as rrun

    app = rrun.with_task(rconfig.load(station_config), HOLD_STILL)
    dream_config = rrun.policy_config(app, checkpoint)
    body = bodies.for_config(app)
    arms: dict = {}
    readers: list = []
    not_off: list[str] = []
    try:
        readers = await asyncio.to_thread(rrun.make_cameras, app)
        kin = rrun.make_kinematics(app)
        client = await asyncio.to_thread(rrun.open_client, dream_config)
        warmup = getattr(rrun, "TRAJ_WARMUP_S", None)  # only some dreamlabs versions compile the IK up front
        if warmup is not None and hasattr(client, "wait_for_ik"):
            if not await asyncio.to_thread(client.wait_for_ik, warmup):
                _log.warning("the IK's traj solve did not compile in time; the arms open anyway")
        if stop.is_set():
            control.run_teardown("close the dream client", client.close)
            return
        # The arms last, so a camera or a model that fails to set up never moves one.
        open_arm = body.open_arm_read_only if dry_run and body.open_arm_read_only is not None else body.open_arm
        for name, arm_cfg in (("left", app.follower_l), ("right", app.follower_r)):
            arm = open_arm(arm_cfg, name)
            arms[name] = followers.FollowerArm(name, rrun.DryRun(arm), home_rad=arm.home_rad) if dry_run else arm
        if not dry_run and not await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S):
            raise RuntimeError("an arm did not reach home before the run")
        for reader in readers:
            reader.start()
        await asyncio.sleep(getattr(rrun, "EXPOSURE_SETTLE_S", 1.0))
        for reader in readers:
            reader.set_exposure_hold(True)
        loop = control.ControlLoop(followers=arms, cameras=readers, client=client, task=HOLD_STILL, kinematics=kin,
                                   inference_log_path=app.inference_log_path, camera_owner=False, recorder=None,
                                   **rrun.loop_settings(app))
        fault: BaseException | None = None
        try:
            async with loop:
                handle._aloop, handle._control = asyncio.get_running_loop(), loop
                runner = asyncio.create_task(loop.run(), name="dream-loop")
                await _first_frame(handle)
                handle.ready.set()
                if on_ready is not None:
                    await on_ready()
                stopped = asyncio.create_task(stop.wait(), name="stop")
                await asyncio.wait({runner, stopped}, return_when=asyncio.FIRST_COMPLETED)
                stopped.cancel()
                if runner.done():
                    runner.result()
                runner.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await runner
        except Exception as exc:  # noqa: BLE001 - the arms still go home below
            fault = exc
        finally:
            handle._control = None
        if not dry_run:
            await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S)
        if fault is not None:
            handle.fault = fault
            raise fault
    finally:
        for reader in readers:
            reader.stop()
        for name, opened in arms.items():
            if not control.run_teardown(f"switch {name} arm off", opened.torque_off):
                not_off.append(name)
    if not_off:
        raise RuntimeError(f"arm {' and '.join(not_off)} may still be powered: its switch-off failed; cut its power")


async def _first_frame(handle: StationHandle, timeout_s: float = 10.0) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if handle.snapshot() is not None:
            return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"camera {OVERHEAD!r} or the arm state sent nothing in {timeout_s:.0f} s")
