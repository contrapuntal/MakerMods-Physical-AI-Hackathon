"""Scores ball detectors on the reviewed test frames (held-out episodes), side by side.

Detectors:
- stock: what the dashboard and --augment use today with the stock model: COCO YOLO "sports ball" boxes named by
  their pixels' colour, plus the HSV round-blob finder (no green_cube class).
- <path to .pt>: a fine-tuned model from `train`, its classes as they are.

Per class: found (recall), right when it says so (precision), at IoU >= 0.3 (balls are ~12 px overhead, where a few
pixels of box error is half the IoU). And the number that matters for Dream-1: how often a green cube is called a
green ball.
"""

from __future__ import annotations

import pathlib

import cv2

from finetune import data
from harness import augment


def _stock(weights: str):
    labeller = data.AutoLabeller(weights)

    def predict(rgb):
        return [(b["cls"], b.get("score", 1.0), b["box"]) for b in labeller(rgb) if b["cls"] != "green_cube"]
    return predict


def _finetuned(weights: str, conf: float):
    yolo = augment.YoloBalls(weights, conf=conf)
    if not yolo.colour_model:
        raise SystemExit(f"{weights} has no colour classes: is it a fine-tuned model from `train`?")

    def predict(rgb):
        return [(n, p, list(b)) for n, p, b in yolo.detect_all(rgb, conf) if p >= conf]
    return predict


def score(predict, frames: list[tuple[str, dict]], root: pathlib.Path, iou: float = 0.3) -> dict:
    per = {c: {"tp": 0, "fp": 0, "fn": 0} for c in data.CLASSES}
    cube_as_ball = 0
    for name, rec in frames:
        rgb = cv2.cvtColor(cv2.imread(str(root / "frames" / name)), cv2.COLOR_BGR2RGB)
        preds = predict(rgb)
        gt = [(b["cls"], b["box"]) for b in rec["boxes"]]
        used = set()
        for cls, _, box in sorted(preds, key=lambda p: -p[1]):
            hit = next((i for i, (gc, gb) in enumerate(gt) if i not in used and gc == cls
                        and data._iou(gb, box) >= iou), None)
            if hit is None:
                per[cls]["fp"] += 1
                if cls == "green_ball" and any(gc == "green_cube" and data._iou(gb, box) >= iou for gc, gb in gt):
                    cube_as_ball += 1
            else:
                used.add(hit)
                per[cls]["tp"] += 1
        for i, (gc, _) in enumerate(gt):
            if i not in used:
                per[gc]["fn"] += 1
    out = {}
    for c, m in per.items():
        n_gt, n_pred = m["tp"] + m["fn"], m["tp"] + m["fp"]
        out[c] = {"found": f"{m['tp']}/{n_gt}" if n_gt else "-",
                  "recall": round(m["tp"] / n_gt, 2) if n_gt else None,
                  "precision": round(m["tp"] / n_pred, 2) if n_pred else None}
    out["green_cube_called_green_ball"] = cube_as_ball
    return out


def evaluate(models: list[str], stock_weights: str, conf: float = 0.25, root: pathlib.Path = data.ROOT,
             split: str = "test") -> dict:
    labels = data.load_labels(root)
    frames = [(n, r) for n, r in sorted(labels.items()) if r.get("split") == split and r.get("status") == "ok"]
    if not frames:
        raise SystemExit(f"no reviewed {split} frames yet: review them first (the page's Test filter)")
    results = {}
    for m in models:
        predict = _stock(stock_weights) if m == "stock" else _finetuned(m, conf)
        results[m] = score(predict, frames, root)
    print(f"{len(frames)} reviewed {split} frames (held-out episodes)\n")
    classes = [c for c in data.CLASSES if c != "green_cube"] + ["green_cube"]
    print(f"{'class':<14}" + "".join(f"{pathlib.Path(m).parent.parent.name if m != 'stock' else 'stock':>24}"
                                     for m in models))
    for c in classes:
        cells = []
        for m in models:
            r = results[m][c]
            cells.append(f"{r['found']:>8} R{_f(r['recall'])} P{_f(r['precision'])}")
        print(f"{c:<14}" + "".join(f"{x:>24}" for x in cells))
    print(f"{'cube->ball':<14}" + "".join(f"{results[m]['green_cube_called_green_ball']:>24}" for m in models))
    return results


def _f(v) -> str:
    return "  - " if v is None else f"{v:.2f}"
