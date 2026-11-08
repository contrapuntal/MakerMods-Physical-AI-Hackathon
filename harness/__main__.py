"""Hill Climb relay harness for NUCK-3.

    # on the station, in ~/team-harness (run with the Dream library's Python):
    ~/dream-env/bin/python -m harness run --color orange --dry-run      # Dream-1 plans, nothing moves
    ~/dream-env/bin/python -m harness run --color orange                # REAL: arms move. Hand on the e-stop.
    ~/dream-env/bin/python -m harness serve [--dry-run]                   # station server for the DimOS harness
    ~/dream-env/bin/python -m harness snapshot --out /tmp/snap          # overhead frame + detections, for calibration
    ~/dream-env/bin/python -m harness detect FRAME.jpg [...]             # detections on saved frames (works anywhere)

`run` takes the match's whole input before "go" (our ball colour from the coin flip) and then runs unattended, as the
rubric requires. Ctrl-C stops it: the arms hold, move home and switch off.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys
import time

from harness.freeze import FreezeControl, StateFile

DEFAULT_CALIBRATION = pathlib.Path(__file__).resolve().parent / "calibration.yaml"  # same as harness.config's
DEFAULT_MODEL = "dreamlabs/dream-1-beta"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the three-stage relay",
                       description="Run the three-stage relay. SIGUSR1 freezes the arms in place (no home, no "
                                   "switch-off); SIGINT/SIGTERM ends the run.")
    r.add_argument("--color", required=True, help="our ping-pong ball colour from the coin flip")
    r.add_argument("--checkpoint", default=DEFAULT_MODEL, help="the Dream model (a fine-tune's id, or base Dream-1)")
    r.add_argument("--dry-run", action="store_true", help="Dream-1 plans; nothing is written to the arms")
    r.add_argument("--unplug-text", default=None,
                   help="the unplug demos' task text, sent word for word (default: the recorded text when "
                        "--checkpoint is a fine-tune, else the rotating base-model wordings)")
    r.add_argument("--time-limit", type=float, default=300.0)
    r.add_argument("--rod-on-tower", action="store_true",
                   help="put the rod in the top cube after it is on the tower (default: on the table, before)")
    r.add_argument("--calibration", type=pathlib.Path, default=DEFAULT_CALIBRATION)
    r.add_argument("--out", type=pathlib.Path, default=None, help="run folder (default ~/hackathon_runs/<time>)")
    s = sub.add_parser("snapshot", help="save the overhead frame with detections drawn on (no arms opened)")
    s.add_argument("--out", type=pathlib.Path, required=True)
    s.add_argument("--calibration", type=pathlib.Path, default=DEFAULT_CALIBRATION)
    v = sub.add_parser("serve", help="hold the station open and serve the relay's steps over HTTP (for DimOS)")
    v.add_argument("--color", default="orange", help="our ball colour (the client's /start can change it)")
    v.add_argument("--checkpoint", default=DEFAULT_MODEL)
    v.add_argument("--dry-run", action="store_true", help="Dream-1 plans; nothing is written to the arms")
    v.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let a laptop on the LAN drive it")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--calibration", type=pathlib.Path, default=DEFAULT_CALIBRATION)
    d = sub.add_parser("detect", help="run detection on saved frames")
    d.add_argument("frames", nargs="+", type=pathlib.Path)
    d.add_argument("--calibration", type=pathlib.Path, default=DEFAULT_CALIBRATION)
    args = ap.parse_args(argv)
    ctl = state = out = None
    if args.cmd == "run":
        # Before the slow imports, so a freeze said while they load is caught and the run aborts before the arms open.
        out = args.out or pathlib.Path("~/hackathon_runs").expanduser() / time.strftime("%Y%m%d-%H%M%S")
        ctl = FreezeControl()
        ctl.install_freeze_signal()
        state = StateFile(out)
        state.update(state="starting", pid=os.getpid(), started_at=time.time(), dry_run=args.dry_run,
                     freeze_supported=True, stage=None, score=None, alarm=None, error=None, version=1)
        state.link_current()
    import cv2

    from harness import vision
    from harness.config import Calibration, MatchConfig
    from harness.relay import TRAINED_UNPLUG

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cal = Calibration.load(args.calibration)

    if args.cmd == "detect":
        for f in args.frames:
            rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            layout = vision.locate_layout(rgb, cal)
            world = vision.detect(rgb, vision.with_layout(cal, layout))
            print(f"{f.name}: layout {layout.note()}")
            out = f.with_name(f.stem + "_detections.jpg")
            cv2.imwrite(str(out), vision.draw(rgb, world))
            print(f"{f}: {world.summary()} -> {out}")
        return

    if args.cmd == "snapshot":
        from robots import config as rconfig, run as rrun
        from harness.dream_executor import OVERHEAD, STATION_CONFIG
        app = rrun.with_task(rconfig.load(STATION_CONFIG), "snapshot")
        readers = rrun.make_cameras(app)
        try:
            for reader in readers:
                reader.start()
            overhead = next(r for r in readers if r.name == OVERHEAD)
            end = time.monotonic() + 10
            while overhead.frame is None and time.monotonic() < end:
                time.sleep(0.05)
            time.sleep(1.0)  # let exposure settle
            frame = overhead.frame
            if frame is None:
                sys.exit("the overhead camera sent no frame")
            args.out.mkdir(parents=True, exist_ok=True)
            rgb = cv2.resize(frame.data, cal.frame_size)
            cv2.imwrite(str(args.out / "overhead.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            layout = vision.locate_layout(rgb, cal)
            print("layout:", layout.note())
            world = vision.detect(rgb, vision.with_layout(cal, layout))
            cv2.imwrite(str(args.out / "overhead_detections.jpg"), vision.draw(rgb, world))
            print(f"{world.summary()}\nwrote {args.out}/overhead.jpg and overhead_detections.jpg")
        finally:
            for reader in readers:
                reader.stop()
        return

    if args.cmd == "serve":
        _serve(args, cal)
        return

    by_spoken = {c.split("_")[0]: c for c in cal.ball_colors}
    if args.color in by_spoken:
        args.color = by_spoken[args.color]  # "--color green" means the green_ball calibration
    if args.color not in cal.ball_colors:
        sys.exit(f"--color {args.color!r} is not one of the ball colours {sorted(by_spoken)}")
    match = MatchConfig(own_color=args.color, time_limit_s=args.time_limit, calibration=cal,
                        rod_first=not args.rod_on_tower,
                        trained_unplug=args.unplug_text or (TRAINED_UNPLUG if args.checkpoint.startswith("dream-1-ft")
                                                            else None))
    from harness.dream_executor import run_match
    relay = asyncio.run(run_match(match, args.checkpoint, dry_run=args.dry_run, out_dir=out, ctl=ctl, state=state))
    print(f"score {relay.score()}  passed {relay.passed}  report: {out}/report.json")


def _serve(args: argparse.Namespace, cal: Calibration) -> None:
    """Opens the station (station.py), then serves StationServer over HTTP until Ctrl-C (arms home, switch off)."""
    import signal
    import threading

    from harness.config import MatchConfig
    from harness.station import StationHandle, run_station
    from harness.station_server import StationServer, make_http

    handle = StationHandle()
    by_spoken = {c.split("_")[0]: c for c in cal.ball_colors}
    color = by_spoken.get(args.color, args.color)
    server = StationServer(handle, MatchConfig(own_color=color, calibration=cal))
    http = make_http(server, args.host, args.port)

    async def serve() -> None:
        stop = asyncio.Event()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            asyncio.get_running_loop().add_signal_handler(signum, stop.set)

        serving = []

        async def on_ready() -> None:
            threading.Thread(target=http.serve_forever, name="station-http", daemon=True).start()
            serving.append(True)
            mode = "DRY RUN, nothing moves" if args.dry_run else "LIVE: arms move on each step"
            print(f"station server on http://{args.host}:{args.port} ({mode}). Ctrl-C stops: arms home, off.")

        try:
            await run_station(handle, args.checkpoint, dry_run=args.dry_run, stop=stop, on_ready=on_ready)
        finally:
            if serving:  # shutdown() waits for serve_forever, so only call it once that started
                http.shutdown()
            http.server_close()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
