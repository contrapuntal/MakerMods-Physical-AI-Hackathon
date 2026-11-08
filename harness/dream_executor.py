"""Runs the relay on the real station with Dream-1: one control loop, the instruction switched between steps.

This mirrors `dreamcli robot run` (`robots.run.main`): open the cameras, the Dream client and the arms the same way,
home, then run Dream's own `ControlLoop`. Beside it runs our supervisor, which reads the loop's newest overhead frame
and arm state twice a second, feeds them to the relay, and calls `loop.set_task()` whenever the relay moves to the
next instruction, so the arms never go home between steps. When the relay is done, the clock runs out, or Ctrl-C is
pressed, the loop stops the way `dreamcli` stops it: the arms hold, move home and switch off.

A dry run (`dry_run=True`) opens the arms read-only and writes no target to them, exactly as `dreamcli ... --dry-run`.
Nothing moves, so steps time out; it checks the wiring, the camera calibration and the instruction sequence.

Only this module imports Dream's `robots` package, so the relay and vision run (and are tested) without it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import pathlib
import signal
import time

import cv2

from harness.claude_check import ClaudeVerifier
from harness.config import MatchConfig
from harness.freeze import FreezeControl, FreezeRequested, first_alarm, hold_measured
from harness.relay import Outcome, Relay
from harness import vision

_log = logging.getLogger(__name__)

STATION_CONFIG = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
OVERHEAD = "scene"  # the overhead camera's name in the station config
POLL_S = 0.5
FROZEN_POLL_S = 0.5  # how often the frozen wait looks for an end request
ALARM_POLL_S = 1.0  # ... and for an arm alarm
VIEW_PORT = 8093  # Dream's DebugView: what the model sees (the page `dreamcli robot run --view` serves)


def _set_state(statefile, /, **fields) -> None:
    """Writes state.json when there is one; a failed write is logged, never allowed to break the run or its teardown."""
    if statefile is None:
        return
    try:
        statefile.update(**fields)
    except Exception:  # noqa: BLE001
        _log.exception("could not write state.json")


async def run_match(match: MatchConfig, checkpoint: str, *, dry_run: bool, out_dir: pathlib.Path,
                    station_config: pathlib.Path = STATION_CONFIG, ctl=None, state=None) -> Relay:
    from robots import config as rconfig
    from robots import control, followers, home, run as rrun
    from robots import bodies
    from robots import errors as rerrors
    from robots import arms_lock
    from robots import types as rtypes
    import dream

    if ctl is None:
        ctl = FreezeControl()

    out_dir.mkdir(parents=True, exist_ok=True)
    # One holder of the arms at a time, as `dreamcli robot run` does; released last, after the arms switch off.
    lock = contextlib.ExitStack()
    try:
        lock.enter_context(arms_lock.holding(pathlib.Path("~/.dreamlabs/arms.lock").expanduser(), "harness run"))
    except Exception as exc:  # noqa: BLE001 - held by dreamcli or the station agent: nothing is opened
        _set_state(state, state="error", error=str(exc))
        raise
    stop = asyncio.Event()
    aloop = asyncio.get_running_loop()
    ctl.on_freeze = lambda: aloop.call_soon_threadsafe(stop.set)  # only wakes the loop: no arm or file work here
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        aloop.add_signal_handler(signum, lambda: (stop.set(), ctl.request_end()))

    relay = Relay(match)
    app = rrun.with_task(rconfig.load(station_config), "hold the arms still")
    dream_config = rrun.policy_config(app, checkpoint)
    body = bodies.for_config(app)
    arms: dict = {}
    readers: list = []
    view = None
    error: str | None = None
    done_error: str | None = None  # a clean abort: written with state "done"
    try:
        readers = await asyncio.to_thread(rrun.make_cameras, app)
        kin = rrun.make_kinematics(app)
        client = await asyncio.to_thread(rrun.open_client, dream_config)
        warmup = getattr(rrun, "TRAJ_WARMUP_S", None)  # only some dreamlabs versions compile the IK up front
        if warmup is not None and hasattr(client, "wait_for_ik"):
            if not await asyncio.to_thread(client.wait_for_ik, warmup):
                _log.warning("the IK's traj solve did not compile in time; the arms open anyway")
        try:
            view = dream.DebugView(client, port=VIEW_PORT)
            _log.info("what the model sees: %s", view.url)
        except OSError as exc:
            _log.warning("debug view cannot listen on port %d (%s); carrying on without it", VIEW_PORT, exc)
        if ctl.frozen:  # a freeze while the harness was starting: the arms are never opened
            done_error = "frozen before the arms opened"
            return relay
        open_arm = body.open_arm_read_only if dry_run and body.open_arm_read_only is not None else body.open_arm
        for name, arm_cfg in (("left", app.follower_l), ("right", app.follower_r)):
            arm = open_arm(arm_cfg, name)
            arms[name] = followers.FollowerArm(name, rrun.DryRun(arm), home_rad=arm.home_rad) if dry_run else arm
        if ctl.frozen:
            raise FreezeRequested  # opened but not homed: hold where they are
        if not dry_run and not await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                                       sleep=ctl.interruptible_sleep):
            raise RuntimeError("an arm did not reach home before the run")
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.set_exposure_hold(True)
        overhead = next(r for r in readers if r.name == OVERHEAD)
        first = await _first_frame(overhead)
        if ctl.frozen:
            raise FreezeRequested
        # Find the bin and the strip where they stand today (the judges place them), then plan from that.
        layout = vision.locate_layout(first.data, match.calibration)
        match.calibration = vision.with_layout(match.calibration, layout)
        relay.cal = match.calibration
        _log.info("layout at go: %s; bin %s, hub %s", layout.note(), match.calibration.bin_roi,
                  match.calibration.hub_roi)
        world = vision.detect(first.data, match.calibration, first.timestamp_s)
        cv2.imwrite(str(out_dir / "go.jpg"), vision.draw(first.data, world))
        t0 = time.monotonic()
        relay.start(world, t0)
        step = relay.next_step(world, t0)
        if step is None:
            raise RuntimeError("the relay has nothing to do at go")
        settings = rrun.loop_settings(app)
        # Dream-1 answered in up to ~4.3 s in practice: don't throw such a chunk away, and ride out a 20 s gap.
        settings["action_timeout_s"] = max(settings.get("action_timeout_s") or 0, 5.0)
        settings["max_stall_ticks"] = max(settings.get("max_stall_ticks") or 0, 400)
        fault: Exception | None = None
        restarts = 0
        _set_state(state, state="running", stage=relay.stage)
        claude = ClaudeVerifier()
        problem = await asyncio.to_thread(claude.probe)
        if problem:
            _log.warning("Claude vision checks off: %s", problem)
        while not stop.is_set():
            task = relay.step.instruction if relay.step else step.instruction
            loop = control.ControlLoop(followers=arms, cameras=readers, client=client, task=task,
                                       kinematics=kin, inference_log_path=app.inference_log_path, camera_owner=False,
                                       recorder=None, **settings)
            try:
                async with loop:
                    runner = asyncio.create_task(loop.run(), name="dream-loop")
                    supervisor = asyncio.create_task(
                        _supervise(loop, relay, match, stop, out_dir, claude, go_rgb=first.data, statefile=state),
                        name="relay")
                    stopped = asyncio.create_task(stop.wait(), name="stop")
                    await asyncio.wait({runner, supervisor, stopped}, return_when=asyncio.FIRST_COMPLETED)
                    for t in (supervisor, stopped):
                        t.cancel()
                    if runner.done():
                        runner.result()
                    runner.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await runner
            except (rerrors.ArmError, rerrors.SafetyStop):
                raise  # a real arm fault or safety stop ends the run (the arms go home in the finally below)
            except rerrors.CameraError as exc:
                # A wrist camera's USB link dropped (it re-enumerates a second or two later). The dead reader stays
                # dead, so reopening only the loop fails at once: close every reader and open them again.
                restarts += 1
                left = relay.time_left(time.monotonic())
                relay.log(time.monotonic(), f"camera dropped ({exc}); reopening cameras, restart {restarts} "
                                            f"with {left:.0f} s left")
                if restarts > 3 or left < match.stop_margin_s + 15:
                    fault = exc
                    break
                readers = await _reopen_cameras(rrun, app, readers, deadline_s=min(12.0, left - match.stop_margin_s))
                if readers is None:
                    fault = exc
                    readers = []
                    break
                continue
            except Exception as exc:  # noqa: BLE001 - a stall, or a fault inside Dream's client (an IK hiccup)
                left = relay.time_left(time.monotonic())
                if "unsafe instruction in this scene" in str(exc):
                    # Dream-1's own safety check: a person is in the scene (a hand on the cable, say). Honour it:
                    # the arms hold, and we wait for the person to step back; every new loop is checked again
                    # by Dream's service, so this only resumes once the scene is clear.
                    relay.log(time.monotonic(), f"Dream refused while a person is in the scene ({exc}); "
                                                f"holding, retry in 3 s with {left:.0f} s left")
                    if left < match.stop_margin_s + 5:
                        break
                    await asyncio.sleep(3.0)
                    continue
                # Dream-1 stopped answering for a while (the loop's stall guard). The arms hold; while the match
                # clock has time left, open a fresh loop and carry on from the same step.
                restarts += 1
                relay.log(time.monotonic(), f"Dream loop stopped ({exc}); restart {restarts} with {left:.0f} s left")
                if restarts > 3 or left < match.stop_margin_s + 10:
                    fault = exc
                    break
                client = await asyncio.to_thread(rrun.open_client, dream_config)
                if view is not None:
                    view.watch(client)  # keep the :8093 page following the new client
                # Warm the new client's IK first, as dreamcli does before the arms move: skipping it crashed the
                # first answer of a restarted loop (SVD did not converge).
                if not await asyncio.to_thread(client.wait_for_ik, rrun.TRAJ_WARMUP_S):
                    _log.warning("restart: the IK's traj solve did not compile in time")
                continue
            break
        if ctl.frozen:
            raise FreezeRequested  # the loop has exited and the arms hold: wait for the end, never go home
        _set_state(state, state="homing")
        if not dry_run:
            await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                    sleep=ctl.interruptible_sleep)
        if fault is not None:
            raise fault
    except FreezeRequested:
        # Never an error: hold the arms, wait for an end request, then home. A freeze must not reach the caller.
        try:
            await _frozen_wait(arms, ctl, state, stop, home, rtypes, dry_run)
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            raise
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        raise
    finally:
        _set_state(state, state="parking")
        for reader in readers:
            reader.stop()
        if view is not None:
            with contextlib.suppress(Exception):
                view.close()
        for name, opened in arms.items():
            control.run_teardown(f"switch {name} arm off", opened.torque_off)
        try:
            _write_report(relay, out_dir)
        except BaseException as exc:
            error = error or str(exc) or type(exc).__name__
            raise
        finally:
            lock.close()
            if error is None:
                try:
                    _set_state(state, state="done", score=int(relay.score()), error=done_error)
                except Exception as exc:  # noqa: BLE001
                    _set_state(state, state="error", error=str(exc))
            else:
                _set_state(state, state="error", error=error)
    return relay


async def _frozen_wait(arms: dict, ctl: FreezeControl, state, stop: asyncio.Event, home, rtypes,
                       dry_run: bool) -> None:
    """Holds the arms where they are until an end is requested, then homes. Returns once the end move is done.

    The arms are commanded their own measured pose once (the driver's hold tick keeps it from there); there is no
    motion and no switch-off here. A fault is written to state.json and the hold continues. A new freeze during
    the end home move comes back here and holds again.
    """
    make_command = lambda positions, gripper: rtypes.ArmCommand(positions_rad=positions, gripper=gripper)  # noqa: E731
    while True:
        failed = await asyncio.to_thread(hold_measured, arms, make_command)
        fields = {"alarm": "; ".join(f"{n}: could not hold ({e})" for n, e in failed.items())} if failed else {}
        _set_state(state, state="frozen", **fields)  # only now: the control loop is closed and the arms hold
        _log.warning("frozen: arms hold the measured pose; waiting for an end request")
        last_alarm = fields.get("alarm")
        next_alarm = time.monotonic() + ALARM_POLL_S
        while not ctl.end_requested.is_set():
            await asyncio.sleep(FROZEN_POLL_S)
            if time.monotonic() >= next_alarm:
                next_alarm = time.monotonic() + ALARM_POLL_S
                try:
                    alarm = await asyncio.to_thread(first_alarm, arms)
                except Exception:  # noqa: BLE001 - a failed check must never drop the hold
                    _log.exception("alarm check failed")
                    alarm = None
                if alarm and alarm != last_alarm:
                    last_alarm = alarm
                    _log.error("alarm while frozen: %s", alarm)
                    _set_state(state, alarm=alarm)
        if not ctl.begin_end():
            continue  # a freeze arrived since the end request: keep holding (arms were not touched)
        _set_state(state, state="homing")
        try:
            if not dry_run:
                await asyncio.to_thread(home.return_to_home, arms, duration_s=home.HOME_DURATION_S,
                                        sleep=ctl.interruptible_sleep)
        except FreezeRequested:
            continue  # frozen again mid-move: hold the pose it reached
        return


async def _reopen_cameras(rrun, app, readers: list, deadline_s: float) -> list | None:
    """Closes the readers and opens them all again, retrying while the dropped camera comes back on USB.
    Returns the started readers, or None when a camera is still missing at the deadline."""
    for reader in readers:
        with contextlib.suppress(Exception):
            reader.stop()
    end = time.monotonic() + max(deadline_s, 2.0)
    while time.monotonic() < end:
        await asyncio.sleep(1.0)
        try:
            fresh = await asyncio.to_thread(rrun.make_cameras, app)
        except Exception as exc:  # noqa: BLE001 - the device node isn't back yet
            _log.info("cameras not back yet: %s", exc)
            continue
        for reader in fresh:
            reader.start()
        for reader in fresh:
            reader.set_exposure_hold(True)
        try:
            for reader in fresh:
                await _first_frame(reader, timeout_s=max(1.0, min(4.0, end - time.monotonic())))
            _log.info("cameras reopened")
            return fresh
        except RuntimeError as exc:
            _log.info("a camera opened but sent nothing yet: %s", exc)
            for reader in fresh:
                with contextlib.suppress(Exception):
                    reader.stop()
    return None


async def _first_frame(reader, timeout_s: float = 10.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if reader.frame is not None:
            return reader.frame
        await asyncio.sleep(0.05)
    raise RuntimeError(f"camera {reader.name!r} sent no frame in {timeout_s:.0f} s")


async def _supervise(loop, relay: Relay, match: MatchConfig, stop: asyncio.Event, out_dir: pathlib.Path,
                     claude: ClaudeVerifier, go_rgb=None, statefile=None) -> None:
    if claude.available:
        relay.confirm_kinds = {"unplug", "hold"}  # the two pass bars that, if wrong, mean a DQ or a lost +20
        _log.info("Claude vision checks on (%s)", "unplug and tower confirmations, 8-ball watch")
    else:
        _log.info("Claude vision checks off (no key); colour checks only")
    pending: dict[str, asyncio.Task] = {}  # question -> running Claude call
    last_unplug_ask = 0.0
    saved = len(list(out_dir.glob("step_*.jpg")))  # carries on numbering after a loop restart
    stage = None  # a restarted supervisor writes the current stage on its first poll
    while not stop.is_set():
        await asyncio.sleep(POLL_S)
        if relay.stage != stage:
            stage = relay.stage
            _set_state(statefile, stage=stage)
        frame = loop.video.get(OVERHEAD)
        state = loop.state
        if frame is None or state is None:
            continue
        now = time.monotonic()
        world = vision.detect(frame.data, match.calibration, frame.timestamp_s)
        # Collect Claude answers that came back since the last poll.
        for question, task in list(pending.items()):
            if not task.done():
                continue
            del pending[question]
            yes, why = task.result() if task.exception() is None else (None, repr(task.exception()))
            if not claude.available:
                relay.confirm_kinds.clear()  # Claude went away mid-run: the colour checks decide from here
            if question in ("confirm",):
                # No opinion (unreachable, out of credit) counts as a yes: the colour check already passed.
                relay.resolve_confirm(yes is not False, why, now)
            elif question == "unplugged_since" and yes:
                if relay.external_pass(world, now, why):
                    _next(loop, relay, world, now, stop)
            elif question == "eight_ball_in_bin" and yes:
                relay.log(now, f"Claude sees the 8-ball in the bin: {why}")
                relay.eight_ball_alert = True
        if claude.available:
            if relay.pending_confirm and "confirm" not in pending:
                if relay.pending_confirm == "unplug":  # compared with go: a cable already out at go doesn't count
                    call = (claude.ask, "unplugged_since", frame.data, go_rgb)
                else:
                    call = (claude.ask, "tower", frame.data)
                pending["confirm"] = asyncio.create_task(asyncio.to_thread(*call))
            step = relay.step
            if step and step.kind == "unplug" and go_rgb is not None and now - last_unplug_ask > 5.0 \
                    and "unplugged_since" not in pending:
                last_unplug_ask = now
                pending["unplugged_since"] = asyncio.create_task(
                    asyncio.to_thread(claude.ask, "unplugged_since", frame.data, go_rgb))
        grip = {"left": float(state.left.gripper), "right": float(state.right.gripper)}
        outcome = relay.observe(world, now, grip)
        if outcome is Outcome.RUNNING:
            continue
        if saved < 200:  # one annotated frame per finished step, for the review afterwards
            cv2.imwrite(str(out_dir / f"step_{saved:03d}_{outcome.value}.jpg"), vision.draw(frame.data, world))
            saved += 1
        finished = relay.step  # None after a pass; the step itself after a failure
        if claude.available and outcome is Outcome.PASSED and relay.stage == "balls" and \
                "eight_ball_in_bin" not in pending:
            pending["eight_ball_in_bin"] = asyncio.create_task(
                asyncio.to_thread(claude.ask, "eight_ball_in_bin", frame.data))
        del finished
        _next(loop, relay, world, now, stop)
    for task in pending.values():
        task.cancel()


def _next(loop, relay: Relay, world, now: float, stop: asyncio.Event) -> None:
    step = relay.next_step(world, now)
    if step is None:
        relay.log(now, f"done: score {relay.score()} (stages passed: {list(relay.passed)})")
        stop.set()
        return
    # Dream starts a fresh episode on every set_task(), even with the same words, dropping the move in progress.
    # A retry of the same instruction therefore just lets Dream-1 carry on.
    if step.instruction != loop.task:
        loop.set_task(step.instruction)


def _write_report(relay: Relay, out_dir: pathlib.Path) -> None:
    report = {
        "score": relay.score(),
        "passed": relay.passed,
        "own_in_bin": relay.own_in_bin,
        "other_in_bin": relay.other_in_bin,
        "tower": relay.placed,
        "events": [{"t_s": round(e.t_s, 2), "stage": e.stage, "what": e.what} for e in relay.events],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
