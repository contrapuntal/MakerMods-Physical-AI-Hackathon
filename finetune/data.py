"""Frames, labels and the YOLO dataset for the ball detector.

Classes (CLASSES): the four ball colours of a match, plus the green cube, the decoy Dream-1 grabs instead of the
green ball. Training the cube as its own class teaches the model the difference rather than just ignoring it.

Layout under ft_data/:
    raw/<dataset>/videos/chunk-000/observation.images.<camera>/episode_NNNNNN.mp4   copied off the station (pull)
    frames/<dataset>__epNNN__<camera>__fNNNNN.jpg                                    one in every --step frames
    labels.json        per frame: {"split", "status": auto|ok|skip, "boxes": [{"cls", "box": [x0,y0,x1,y1], "src"}]}
    yolo/              the exported dataset (images/, labels/, balls.yaml), rebuilt by `export`
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

import cv2
import numpy as np
import numpy.typing as npt

from harness import augment
from harness.config import DEFAULT_CALIBRATION, Calibration
from harness.vision import mask_for

ROOT = pathlib.Path(__file__).resolve().parent.parent / "ft_data"
# Appended, never reordered: a class's index is what the YOLO label files store.
CLASSES = ("green_ball", "orange_ball", "white_ball", "black_ball", "green_cube",
           "blue_ball", "yellow_ball", "rod", "cube")  # rod: any colour; cube: blue or purple (green_cube is its own)
# Calibration colour of each ball class (augment.ball_color's answers), and back.
CAL_COLOR = {"green_ball": "green_ball", "orange_ball": "orange", "white_ball": "white", "black_ball": "black",
             "blue_ball": "blue_ball", "yellow_ball": "yellow_ball"}
CLASS_OF = {v: k for k, v in CAL_COLOR.items()}
STATION = "pah@10.103.0.125"
_NAME = re.compile(r"^(?P<ds>.+)__ep(?P<ep>\d+)__(?P<cam>[a-z_]+)__f(?P<f>\d+)\.jpg$")


def frame_info(name: str) -> dict:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a frame name: {name}")
    return {"dataset": m["ds"], "episode": int(m["ep"]), "camera": m["cam"], "frame": int(m["f"])}


# ---------------------------------------------------------------------- pull + extract
def pull(dataset: str, root: pathlib.Path = ROOT, station: str = STATION) -> pathlib.Path:
    """Copies one dataset's videos and metadata off the station (rsync over ssh; only what is new)."""
    dst = root / "raw"
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include=*/", "--include=*.mp4", "--include=*.jsonl", "--include=info.json",
                    "--exclude=*", f"{station}:hackathon_data/{dataset}", str(dst) + "/"], check=True)
    return dst / dataset


def split_of(episode: int, n_episodes: int, val: int, test: int) -> str:
    """Whole episodes per split, the last ones held out: frames of one episode are near-duplicates, so a frame-level
    split would score the model on pictures it trained on."""
    if episode >= n_episodes - test:
        return "test"
    if episode >= n_episodes - test - val:
        return "val"
    return "train"


def extract(dataset: str, step: int = 20, val: int = 1, test: int = 3, root: pathlib.Path = ROOT) -> int:
    """Saves one frame in every `step` of every camera of every episode; records each frame's split in labels.json."""
    videos = sorted((root / "raw" / dataset / "videos").glob("chunk-*/observation.images.*/episode_*.mp4"))
    if not videos:
        raise SystemExit(f"no videos under {root / 'raw' / dataset}: run `pull {dataset}` first")
    episodes = sorted({int(v.stem.split("_")[1]) for v in videos})
    out = root / "frames"
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(root)
    n = 0
    for v in videos:
        ep = int(v.stem.split("_")[1])
        cam = v.parent.name.split(".")[-1]
        split = split_of(episodes.index(ep), len(episodes), val, test)
        cap = cv2.VideoCapture(str(v))
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                name = f"{dataset}__ep{ep:03d}__{cam}__f{i:05d}.jpg"
                if not (out / name).exists():
                    cv2.imwrite(str(out / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                labels.setdefault(name, {"status": "new", "boxes": []})["split"] = split
                n += 1
            i += 1
    save_labels(labels, root)
    return n


# ---------------------------------------------------------------------- labels.json
def load_labels(root: pathlib.Path = ROOT) -> dict:
    p = root / "labels.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_labels(labels: dict, root: pathlib.Path = ROOT) -> None:
    p = root / "labels.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=0, sort_keys=True))
    os.replace(tmp, p)  # atomic: the review page and the labeller never see half a file


# ---------------------------------------------------------------------- auto-labelling
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def find_cubes(rgb: npt.NDArray[np.uint8], cal: Calibration, color: str = "green") -> list[list[float]]:
    """Squarish, solid blobs of a cube colour at any scale: candidate cube boxes."""
    h, w = rgb.shape[:2]
    mask = mask_for(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), cal.colors[color])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if not 4e-4 * w * h <= area <= 0.2 * w * h:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if max(rw, rh) / max(1.0, min(rw, rh)) > 1.8 or hull <= 0 or area / hull < 0.8:
            continue  # rods are long; the cube's hole leaves the outline solid
        x, y, bw, bh = cv2.boundingRect(c)
        out.append([float(x), float(y), float(x + bw), float(y + bh)])
    return out


def find_rods(rgb: npt.NDArray[np.uint8], cal: Calibration) -> list[list[float]]:
    """Long, solid blobs of a rod colour (red, green, yellow): candidate rod boxes, lying or standing."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out = []
    for color in cal.rod_colors:
        mask = mask_for(hsv, cal.colors[color])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(c)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            hull = cv2.contourArea(cv2.convexHull(c))
            if not 4e-4 * w * h <= area <= 0.2 * w * h or hull <= 0 or area / hull < 0.75:
                continue
            if max(rw, rh) / max(1.0, min(rw, rh)) >= 2.2:  # a rod is long; a cube or ball is not
                x, y, bw, bh = cv2.boundingRect(c)
                out.append([float(x), float(y), float(x + bw), float(y + bh)])
    return out


class AutoLabeller:
    """First-pass boxes from the detectors the harness already has: YOLO's sports balls named by colour, the HSV
    round-blob finder per ball colour, and HSV square blobs for the green cube. A person reviews every frame."""

    def __init__(self, weights: str, cal: Calibration | None = None) -> None:
        self.cal = cal or Calibration.load(DEFAULT_CALIBRATION)
        self.yolo = augment.YoloBalls(weights, conf=0.05)

    def __call__(self, rgb: npt.NDArray[np.uint8]) -> list[dict]:
        boxes: list[dict] = []

        def add(cls: str, box, src: str, score: float) -> None:
            for b in boxes:
                if _iou(b["box"], box) > 0.3:
                    if b["src"] == "hsv" and src == "yolo":  # YOLO's box is the tighter fit
                        b.update(box=[round(v, 1) for v in box], src="yolo+hsv", score=score)
                    elif b["src"] == "yolo" and src == "hsv":
                        b["src"] = "yolo+hsv"
                    return
            boxes.append({"cls": cls, "box": [round(v, 1) for v in box], "src": src, "score": round(score, 2)})

        for color, p, box in augment.label_balls(rgb, self.yolo.detect_all(rgb), self.cal, keep_white=True):
            if color in CLASS_OF and p >= 0.08:
                add(CLASS_OF[color], box, "yolo", p)
        for color in ("green_ball", "orange", "white", *self.cal.other_balls):
            for b in augment.find_hsv(rgb, color, self.cal):
                add(CLASS_OF[color], [b.x - b.r, b.y - b.r, b.x + b.r, b.y + b.r], "hsv", b.score)
        decoys = [("green_cube", box) for box in find_cubes(rgb, self.cal)]
        decoys += [("cube", box) for c in ("blue", "purple") for box in find_cubes(rgb, self.cal, c)]
        decoys += [("rod", box) for box in find_rods(rgb, self.cal)]
        for cls, box in decoys:
            if not any(_iou(b["box"], box) > 0.3 for b in boxes):
                boxes.append({"cls": cls, "box": [round(v, 1) for v in box], "src": "hsv", "score": 0.5})
        return boxes


def autolabel(weights: str, root: pathlib.Path = ROOT, redo: bool = False) -> int:
    """Fills in boxes for every frame still "new" (or every unreviewed frame with `redo`). Reviewed frames are kept."""
    labels = load_labels(root)
    labeller = AutoLabeller(weights)
    n = 0
    for name, rec in sorted(labels.items()):
        if rec.get("status") in ("ok", "skip") or (rec.get("status") == "auto" and not redo):
            continue
        bgr = cv2.imread(str(root / "frames" / name))
        if bgr is None:
            continue
        rec["boxes"] = labeller(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rec["status"] = "auto"
        n += 1
        if n % 100 == 0:
            save_labels(labels, root)
            print(f"  {n} frames labelled")
    save_labels(labels, root)
    return n


# ---------------------------------------------------------------------- export
def export(root: pathlib.Path = ROOT, include_auto: bool = False) -> pathlib.Path:
    """Writes ft_data/yolo/ (images + YOLO txt labels per split) and balls.yaml from the reviewed frames.

    `include_auto` also takes frames nobody reviewed yet (their labels are the detectors' guesses: a quick first model
    only; the test split is always reviewed frames alone, so the score stays honest)."""
    labels = load_labels(root)
    out = root / "yolo"
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, int] = {}
    for name, rec in sorted(labels.items()):
        split = rec.get("split", "train")
        ok = rec.get("status") == "ok" or (include_auto and rec.get("status") == "auto" and split != "test")
        if not ok:
            continue
        src = root / "frames" / name
        h, w = cv2.imread(str(src)).shape[:2]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out / "images" / split / name)
        lines = []
        for b in rec["boxes"]:
            if rec.get("status") == "auto" and b["cls"] in ("green_cube", "cube", "rod"):
                continue  # the first pass's decoy boxes were mostly wrong in review (plugs, shadows): never train on them
            x0, y0, x1, y1 = b["box"]
            lines.append(f"{CLASSES.index(b['cls'])} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        (out / "labels" / split / name).with_suffix(".txt").write_text("\n".join(lines))
        counts[split] = counts.get(split, 0) + 1
    if not counts.get("train") or not counts.get("val"):
        raise SystemExit(f"not enough reviewed frames to train ({counts or 'none'}): review some train and val "
                         "frames first, or pass --include-auto")
    yaml = out / "balls.yaml"
    yaml.write_text(f"path: {out}\ntrain: images/train\nval: images/val\n"
                    + ("test: images/test\n" if counts.get("test") else "")
                    + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"exported {counts} -> {yaml}")
    return yaml


def stats(root: pathlib.Path = ROOT) -> dict:
    labels = load_labels(root)
    by: dict[str, dict[str, int]] = {}
    boxes: dict[str, int] = {c: 0 for c in CLASSES}
    for rec in labels.values():
        s = by.setdefault(rec.get("split", "?"), {})
        s[rec.get("status", "?")] = s.get(rec.get("status", "?"), 0) + 1
        if rec.get("status") == "ok":
            for b in rec["boxes"]:
                boxes[b["cls"]] = boxes.get(b["cls"], 0) + 1
    return {"frames": len(labels), "by_split": by, "reviewed_boxes": boxes}
