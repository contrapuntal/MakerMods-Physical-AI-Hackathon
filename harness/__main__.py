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
import pathlib
import sys
import time

import cv2

from harness import vision
from harness.config import DEFAULT_CALIBRATION, Calibration, MatchConfig
from harness.relay import TRAINED_BALLS, TRAINED_STACK, TRAINED_UNPLUG

DEFAULT_MODEL = "dreamlabs/dream-1-beta"
STAGES = ("unplug", "stack", "balls")
TRAINED_TEXTS = {"unplug": TRAINED_UNPLUG, "stack": TRAINED_STACK, "balls": TRAINED_BALLS}

# The team's fine-tunes (all finished 2026-11-08), each with the exact sentence its demos were recorded with.
MODEL_TEXTS = {
    "dream-1-ft:39478655-nuck3-usb-cables": "Remove Orange Cables from their port",
    "dream-1-ft:39478655-nuck3-stack-blocks": "Stack the green, purple, and blue blocks",
    "dream-1-ft:39478655-nuck3-balls-whites":
        "Place all yellow, orange, green, and blue balls in the basket. Push away all white balls.",
    "dream-1-ft:39478655-nuck3-balls-basket": "Place yellow, blue, green, and orange balls into basket.",
    "dream-1-ft:39478655-nuck3-white-balls": "Move nearby white balls far away",
}
# A run with no --checkpoint and no --model-<stage> uses these: one fine-tune per stage.
DEFAULT_STAGE_MODELS = {
    "unplug": "dream-1-ft:39478655-nuck3-usb-cables",
    "stack": "dream-1-ft:39478655-nuck3-stack-blocks",
    "balls": "dream-1-ft:39478655-nuck3-balls-basket",  # coloured balls into the basket; no white-ball pushing
}


def plan_stages(checkpoint: str | None, models: dict[str, str | None],
                texts: dict[str, str | None]) -> tuple[dict[str, str], dict[str, str]]:
    """Each stage's model and its word-for-word task text.

    A stage without its own model runs on `checkpoint`, or, with no checkpoint either, on the team's fine-tune
    for it (DEFAULT_STAGE_MODELS). A stage on a fine-tune (`dream-1-ft:...`) gets that model's recorded sentence
    unless one is given ("" turns it off: the step-by-step base-model plan); a stage on base Dream-1 gets a text
    only when one is given."""
    stage_models = {st: models.get(st) or checkpoint or DEFAULT_STAGE_MODELS[st] for st in STAGES}
    stage_texts = {}
    for st in STAGES:
        given, model = texts.get(st), stage_models[st]
        text = given if given is not None else (MODEL_TEXTS.get(model) or TRAINED_TEXTS[st]
                                                if model.startswith("dream-1-ft") else None)
        if text:
            stage_texts[st] = text
    return stage_models, stage_texts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="harness", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the three-stage relay")
    r.add_argument("--color", required=True, help="our ping-pong ball colour from the coin flip")
    r.add_argument("--checkpoint", default=None,
                   help="one Dream model for every stage (an all-in-one fine-tune, or base Dream-1: "
                        f"{DEFAULT_MODEL}); default: the team's fine-tune per stage")
    r.add_argument("--dry-run", action="store_true", help="Dream-1 plans; nothing is written to the arms")
    r.add_argument("--start-stage", choices=STAGES, default="unplug",
                   help="practice: begin at this stage (skips the earlier ones); a real match starts at unplug")
    r.add_argument("--pull-cables-until", type=float, default=0.0, metavar="S",
                   help="keep the cable model pulling cables until S seconds into the match once one is out "
                        "(0 = move on at the first cable)")
    r.add_argument("--stack-pass", choices=("6in", "3-cubes"), default="6in",
                   help="what Claude must see to pass a whole-tower stack: a 6 in tower (default) or 3 cubes stacked")
    r.add_argument("--all-colors", action="store_true",
                   help="every ball colour but white and black scores (the ball demos' rule); default: only --color")
    r.add_argument("--gravity-scale", type=float, default=1.0,
                   help="Metal gravity compensation x this (0 = off, as the demos were recorded; 1 = Dream's own); "
                        "see harness/gravity.py. Don't combine 0 with --lower-<stage>-cm")
    for st in STAGES:
        r.add_argument(f"--gravity-{st}", type=float, default=None,
                       help=f"gravity compensation while the {st} stage runs (default: --gravity-scale)")
        r.add_argument(f"--model-{st}", default=None,
                       help=f"the Dream model for the {st} stage (default: --checkpoint); the harness switches "
                            "models between stages")
        r.add_argument(f"--lower-{st}-cm", type=float, default=0.0,
                       help=f"lower the {st} model's right-hand targets near the table by this many cm (0 = off; "
                            "up to 10; never below 4 cm; see harness/height_offset.py)")
        r.add_argument(f"--{st}-text", default=None,
                       help=f"the {st} demos' task text, sent word for word (default: {TRAINED_TEXTS[st]!r} when "
                            f"the stage's model is a fine-tune; \"\" for the step-by-step base-model plan)")
    r.add_argument("--time-limit", type=float, default=300.0)
    r.add_argument("--rod-on-tower", action="store_true",
                   help="put the rod in the top cube after it is on the tower (default: on the table, before)")
    r.add_argument("--calibration", type=pathlib.Path, default=DEFAULT_CALIBRATION)
    r.add_argument("--out", type=pathlib.Path, default=None, help="run folder (default ~/hackathon_runs/<time>)")
    _augment_args(r)
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
    _augment_args(v)
    d = sub.add_parser("detect", help="run detection on saved frames")
    d.add_argument("frames", nargs="+", type=pathlib.Path)
    d.add_argument("--calibration", type=pathlib.Path, default=DEFAULT_CALIBRATION)
    args = ap.parse_args(argv)
    from harness import logcolor
    logcolor.install("%(asctime)s %(levelname)s %(name)s: %(message)s")
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
    stage_models, stage_texts = plan_stages(
        args.checkpoint, {st: getattr(args, f"model_{st}") for st in STAGES},
        {st: getattr(args, f"{st}_text") for st in STAGES})
    for st in STAGES:
        print(f"{st:>6}: {stage_models[st]}  text: {stage_texts.get(st, '(step-by-step base-model plan)')!r}")
    from harness import height_offset
    stage_gravity = {st: getattr(args, f"gravity_{st}") for st in STAGES if getattr(args, f"gravity_{st}") is not None}
    if args.gravity_scale != 1.0 or stage_gravity:
        from harness import gravity
        gravity.install(stage_gravity.get("unplug", args.gravity_scale))
        stage_gravity = {st: stage_gravity.get(st, args.gravity_scale) for st in STAGES}
        print("gravity compensation per stage:", {st: f"x{v:g}" for st, v in stage_gravity.items()})
        if min(stage_gravity.values()) < 0.5 and any(getattr(args, f"lower_{st}_cm") for st in STAGES):
            print("WARNING: --lower-*-cm with gravity compensation off lowers the grasp twice")
    for st in STAGES:
        cm = getattr(args, f"lower_{st}_cm")
        if cm:
            height_offset.install(stage_models[st], height_offset.Drop(cm / 100.0))
            print(f"{st:>6}: right-hand targets lowered by up to {cm:.1f} cm near the table (floor 4 cm)")
    match = MatchConfig(own_color=args.color, time_limit_s=args.time_limit, calibration=cal,
                        rod_first=not args.rod_on_tower, stage_models=stage_models, stage_texts=stage_texts,
                        stack_pass=args.stack_pass, all_colors=args.all_colors,
                        pull_cables_until_s=args.pull_cables_until, stage_gravity=stage_gravity,
                        start_stage=args.start_stage)
    out = args.out or pathlib.Path("~/hackathon_runs").expanduser() / time.strftime("%H%M%S")
    from harness.dream_executor import run_match
    relay = asyncio.run(run_match(match, args.checkpoint or stage_models["unplug"], dry_run=args.dry_run, out_dir=out,
                                  augment_spec=args.augment, augment_detector=args.detector))
    print(f"score {relay.score()}  passed {relay.passed}  report: {out}/report.json")
    if height_offset.stats():
        print("height correction:", {k: round(v, 1) for k, v in height_offset.stats().items()})


def _augment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--augment", default="off",
                   help="mark the target ball in the frames sent to Dream: off, or modes box,spotlight,suppress "
                        "(comma-separated; see harness/augment.py)")
    p.add_argument("--detector", default="both", choices=("hsv", "yolo", "both"),
                   help="how --augment finds the ball (yolo needs ultralytics)")


def _serve(args: argparse.Namespace, cal: Calibration) -> None:
    """Opens the station (station.py), then serves StationServer over HTTP until Ctrl-C (arms home, switch off)."""
    import signal
    import threading

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
            await run_station(handle, args.checkpoint, dry_run=args.dry_run, stop=stop, on_ready=on_ready,
                              augment=(cal, args.augment, args.detector))
        finally:
            if serving:  # shutdown() waits for serve_forever, so only call it once that started
                http.shutdown()
            http.server_close()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
