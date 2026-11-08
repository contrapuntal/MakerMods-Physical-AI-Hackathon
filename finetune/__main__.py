"""Fine-tunes YOLO to find the match's balls by colour (and the green cube decoy), from the station's own episodes.

    PY=repos/dimos/.venv/bin/python
    $PY -m finetune pull Remove_White_Balls              # copy the episode videos off the station (rsync over ssh)
    $PY -m finetune extract Remove_White_Balls --step 30  # one frame in 30 per camera; last episodes held out
    $PY -m finetune label                                 # first-pass boxes from the harness's own detectors
    $PY -m finetune review                                # fix them: http://127.0.0.1:8790/ (Test frames first)
    $PY -m finetune train --epochs 60                     # export the reviewed frames, train on this Mac's GPU
    $PY -m finetune eval runs/balls/<run>/weights/best.pt # stock vs fine-tuned on the held-out episodes
    $PY -m finetune stats                                 # how far the review has got

Then use it: HILLCLIMB_YOLO=runs/balls/<run>/weights/best.pt for the dashboard, and the same file for the station's
--augment (harness/augment.py reads its colour classes directly). Data lives in ft_data/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STOCK = str(pathlib.Path("~/dev/hackathon/physical-ai-workshop/yolo11n.pt").expanduser())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finetune", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull", help="copy a dataset's videos off the station")
    p.add_argument("dataset")
    p.add_argument("--station", default="pah@10.103.0.125")
    e = sub.add_parser("extract", help="save frames from the copied videos")
    e.add_argument("dataset")
    e.add_argument("--step", type=int, default=30, help="keep one frame in this many (30 fps video)")
    e.add_argument("--val", type=int, default=1, help="episodes held out for validation (before the test ones)")
    e.add_argument("--test", type=int, default=3, help="last episodes held out for the score")
    lb = sub.add_parser("label", help="first-pass boxes for the frames not labelled yet")
    lb.add_argument("--weights", default=STOCK, help="the stock YOLO the first pass uses")
    lb.add_argument("--redo", action="store_true", help="relabel every frame not reviewed yet")
    r = sub.add_parser("review", help="the review page")
    r.add_argument("--port", type=int, default=8790)
    t = sub.add_parser("train", help="export the reviewed frames and fine-tune")
    t.add_argument("--weights", default=STOCK, help="the model to start from")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imgsz", type=int, default=640, help="960 helps the ~12 px balls overhead; slower")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--name", default=None, help="run name under runs/balls/ (default: a numbered one)")
    t.add_argument("--include-auto", action="store_true",
                   help="also train on unreviewed frames' first-pass boxes (a quick first model; test stays reviewed)")
    t.add_argument("--device", default=None, help="mps (this Mac's GPU, the default when present), cpu, 0 ...")
    v = sub.add_parser("eval", help="score detectors on the reviewed held-out frames")
    v.add_argument("models", nargs="*", help="fine-tuned .pt files; 'stock' (always included) is today's pipeline")
    v.add_argument("--conf", type=float, default=0.25)
    v.add_argument("--split", default="test")
    v.add_argument("--stock-weights", default=STOCK)
    sub.add_parser("stats", help="review progress")
    args = ap.parse_args(argv)

    from finetune import data

    if args.cmd == "pull":
        print(f"copied to {data.pull(args.dataset, station=args.station)}")
    elif args.cmd == "extract":
        n = data.extract(args.dataset, args.step, args.val, args.test)
        print(f"{n} frames from {args.dataset} in {data.ROOT / 'frames'}")
        print(json.dumps(data.stats(), indent=1))
    elif args.cmd == "label":
        print(f"labelled {data.autolabel(args.weights, redo=args.redo)} frames")
        print(json.dumps(data.stats(), indent=1))
    elif args.cmd == "review":
        from finetune import review
        review.serve(port=args.port)
    elif args.cmd == "train":
        _train(args, data)
    elif args.cmd == "eval":
        from finetune import evaluate
        evaluate.evaluate(["stock", *args.models], args.stock_weights, args.conf, split=args.split)
    elif args.cmd == "stats":
        print(json.dumps(data.stats(), indent=1))


def _train(args: argparse.Namespace, data) -> None:
    import torch
    from ultralytics import YOLO

    yaml = data.export(include_auto=args.include_auto)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(args.weights)
    # Light augmentation: the station's cameras and lighting don't change, but the arms cover the balls, so keep
    # mosaic and scale jitter. No hue shift: hue IS the class (green vs orange vs the green cube).
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
                project=str(ROOT / "runs" / "balls"), name=args.name, exist_ok=bool(args.name),
                hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, fliplr=0.5, mosaic=1.0, scale=0.4, patience=20, plots=True)
    best = pathlib.Path(model.trainer.best)
    print(f"\nbest weights: {best}\nscore it:   repos/dimos/.venv/bin/python -m finetune eval {best}\n"
          f"use it:     HILLCLIMB_YOLO={best} ... (dashboard), and copy it to the station for --augment")


if __name__ == "__main__":
    sys.exit(main())
