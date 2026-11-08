"""Augments what Dream-1 sees: marks the target ball in every camera frame before the frame leaves for Dream.

Dream's `ControlLoop` hands each camera tick to `client.set_observation(frames, observation, timestamp)`, with
`frames` one base64 JPEG per `video.<role>` and `observation["prompt"]` the instruction. `AugmentingClient` wraps the
client: it decodes each frame, finds the target, draws on it, re-encodes it, and passes everything else through
untouched. The target comes from the prompt itself ("pick up the green ping-pong ball ..." -> `green_ball`), so it
switches on and off with the relay's steps without any plumbing; on every other step frames go out as they came.

Modes (combine with commas, e.g. "suppress,box"):
- box:       a thick magenta ring around each target ball.
- spotlight: everything outside the target balls darkened and desaturated.
- suppress:  the distractors (same spoken colour, other shape: the green cube and rod for a green ball) turned grey.

Detectors:
- hsv:  the calibration's HSV range for the target, kept when the blob is round; no extra dependencies.
- yolo: COCO "sports ball" boxes from ultralytics, kept when the box's centre is mostly the target colour, so a green
        cube (not a ball) and an orange ball (not green) are both rejected. Needs `pip install ultralytics`.
- both: the union of the two (the default when ultralytics is installed).

Detection runs on a worker thread per camera, so `set_observation` never waits on a model: each frame is drawn with the
newest detections (at most `max_age_s` old). Nothing here imports Dream, so it is tested and previewed anywhere:

    python -m harness.augment FRAME.jpg [...] --target green_ball --mode box,suppress --detector both
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import math
import pathlib
import re
import threading
import time
from typing import Any, Mapping

import cv2
import numpy as np
import numpy.typing as npt

from harness.config import Calibration, spoken
from harness.vision import mask_for

_log = logging.getLogger(__name__)

MODES = ("box", "spotlight", "suppress")
DETECTORS = ("hsv", "yolo", "both")
RING_BGR = (255, 0, 255)  # magenta: nothing on the table is magenta
SPORTS_BALL = 32  # COCO class id
JPEG_QUALITY = 90
# "pick up the green ping-pong ball", "the green golf ball", "green ball": the colour word before "ball".
_TARGET_RE = re.compile(r"pick up the (\w+) (?:[\w-]+ )?ball")


@dataclasses.dataclass(frozen=True)
class Ball:
    x: float  # centre, px of the frame it was found in
    y: float
    r: float  # radius, px
    score: float = 1.0
    source: str = "hsv"


def target_from_prompt(prompt: str, cal: Calibration) -> str | None:
    """The calibration colour the instruction asks to pick up, or None when it is not a pick-up-a-ball step."""
    m = _TARGET_RE.search(prompt.lower())
    if not m:
        return None
    word = m.group(1)
    for color in cal.ball_colors:
        if spoken(color) == word:
            return color
    return None


def distractor_colors(target: str, cal: Calibration) -> list[str]:
    """Colours of the other-shaped pieces that share the target's spoken colour (the green cube and rod)."""
    word = spoken(target)
    return sorted({c for c in (*cal.cube_colors, *cal.rod_colors) if spoken(c) == word and c != target})


# ---------------------------------------------------------------------- detectors
def find_hsv(rgb: npt.NDArray[np.uint8], target: str, cal: Calibration) -> list[Ball]:
    """Round blobs of the target's HSV range, at any scale (overhead or wrist)."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = mask_for(hsv, cal.colors[target])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lo, hi = 1.5e-4 * w * h, 0.25 * w * h  # a far ball overhead to one filling a wrist view
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if not lo <= area <= hi:
            continue
        roundness = 4 * math.pi * area / max(1.0, cv2.arcLength(c, True)) ** 2
        (_, _), (bw, bh), _ = cv2.minAreaRect(c)
        if roundness < 0.5 or max(bw, bh) / max(1.0, min(bw, bh)) > 2.0:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        out.append(Ball(cx, cy, r, roundness, "hsv"))
    return out


# A fine-tuned model (finetune/) names each ball's colour itself: class name -> calibration colour.
COLOR_CLASSES = {"green_ball": "green_ball", "orange_ball": "orange", "white_ball": "white", "black_ball": "black",
                 "blue_ball": "blue_ball", "yellow_ball": "yellow_ball"}
# A fine-tuned model's decoy classes: drawn and listed for debugging, never a target.
DECOY_CLASSES = ("green_cube", "cube", "rod")

# A fine-tuned model is only trusted from this confidence up. On the held-out episodes (v2, Nov 7) it kept 98%
# precision on green balls at 0.4; at 0.28 it called the green rod (never a class) a green ball on the live feed.
COLOUR_MODEL_CONF = 0.4


class YoloBalls:
    """Ball boxes from YOLO. The stock COCO model: 'sports ball' boxes, kept when the inner disc is mostly the target
    colour. A fine-tuned model with colour classes (COLOR_CLASSES): the target colour's own class, as is."""

    # 640, not 320: a 40 mm ball is ~12 px across in a 640 px overhead frame, and at 320 YOLO never finds it.
    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.10, imgsz: int = 640,
                 min_color_frac: float = 0.25) -> None:
        from ultralytics import YOLO  # imported here so the rest runs without it

        self.model = YOLO(weights)
        self.conf, self.imgsz, self.min_color_frac = conf, imgsz, min_color_frac
        self._lock = threading.Lock()  # one model shared by every camera's worker
        names = self.model.names  # {id: name}
        self.colour_model = bool(set(names.values()) & set(COLOR_CLASSES))
        # The classes asked for: every class of a fine-tuned model (balls and the decoy cube); COCO's sports ball.
        self._classes = None if self.colour_model else [SPORTS_BALL]

    def detect_all(self, rgb: npt.NDArray[np.uint8], conf: float = 0.05) -> list[tuple[str, float, tuple]]:
        """Every ball box YOLO sees, down to `conf`: (class name, confidence, (x0, y0, x1, y1)). The stock model is
        asked for "sports ball" alone, so people, laptops and chairs behind the table never come back (and it runs
        faster); a fine-tuned model answers in its own classes (green_ball, ..., green_cube)."""
        with self._lock:
            res = self.model.predict(rgb[:, :, ::-1], conf=min(conf, self.conf), imgsz=self.imgsz,
                                     classes=self._classes, verbose=False)[0]
        names = res.names
        return [(names[int(c)], float(p), tuple(b)) for c, p, b in
                zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]

    def __call__(self, rgb: npt.NDArray[np.uint8], target: str, cal: Calibration,
                 boxes: list[tuple[str, float, tuple]] | None = None) -> list[Ball]:
        if boxes is None:
            boxes = self.detect_all(rgb, self.conf)
        if self.colour_model:  # the model names the colour: no pixel check
            want = next((k for k, v in COLOR_CLASSES.items() if v == target), None)
            conf = max(self.conf, COLOUR_MODEL_CONF)
            return [Ball((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) / 2, float(p), "yolo")
                    for name, p, b in boxes if name == want and p >= conf]
        balls = [(b, p) for name, p, b in boxes if name == "sports ball" and p >= self.conf]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        color = mask_for(hsv, cal.colors[target])
        out = []
        for (x0, y0, x1, y1), score in balls:
            cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
            disc = np.zeros(color.shape, np.uint8)
            cv2.circle(disc, (int(cx), int(cy)), max(1, int(0.7 * r)), 255, -1)
            inside = cv2.countNonZero(disc)
            if inside and cv2.countNonZero(color & disc) / inside >= self.min_color_frac:
                out.append(Ball(cx, cy, r, float(score), "yolo"))
        return out


def merge(*groups: list[Ball]) -> list[Ball]:
    """The union, a ball found by both kept once (the YOLO one, whose circle is the tighter fit)."""
    kept: list[Ball] = []
    for b in sorted((b for g in groups for b in g), key=lambda b: b.source != "yolo"):
        if all(math.hypot(b.x - k.x, b.y - k.y) > max(b.r, k.r) for k in kept):
            kept.append(b)
    return kept


class Finder:
    """Finds the target balls with the chosen detector(s)."""

    def __init__(self, cal: Calibration, detector: str = "both", yolo_weights: str = "yolo11n.pt") -> None:
        if detector not in DETECTORS:
            raise ValueError(f"detector {detector!r} is not one of {DETECTORS}")
        self.cal = cal
        self.yolo: YoloBalls | None = None
        if detector in ("yolo", "both"):
            try:
                self.yolo = YoloBalls(yolo_weights)
            except ImportError:
                if detector == "yolo":
                    raise
                _log.warning("ultralytics is not installed: the augmentation finds balls by colour only")
        self.use_hsv = detector in ("hsv", "both")

    def __call__(self, rgb: npt.NDArray[np.uint8], target: str,
                 yolo_boxes: list[tuple[str, float, tuple]] | None = None) -> list[Ball]:
        """The target balls. Pass `yolo_boxes` (YoloBalls.detect_all) to reuse one YOLO pass for a debug view."""
        groups = []
        if self.yolo is not None:
            groups.append(self.yolo(rgb, target, self.cal, yolo_boxes))
        if self.use_hsv:
            groups.append(find_hsv(rgb, target, self.cal))
        return merge(*groups)


# ---------------------------------------------------------------------- drawing
def annotate(rgb: npt.NDArray[np.uint8], balls: list[Ball], modes: tuple[str, ...], target: str,
             cal: Calibration) -> npt.NDArray[np.uint8]:
    """The frame (RGB) with the modes applied. A frame with no ball found is changed only by `suppress`."""
    img = rgb.copy()
    h, w = img.shape[:2]
    if "suppress" in modes:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        grey = np.zeros((h, w), np.uint8)
        for color in distractor_colors(target, cal):
            grey |= mask_for(hsv, cal.colors[color])
        for b in balls:  # never grey the target itself, even where the ranges overlap
            cv2.circle(grey, (int(b.x), int(b.y)), int(b.r * 1.2) + 1, 0, -1)
        grey = cv2.dilate(grey, np.ones((3, 3), np.uint8))
        hsv[:, :, 1][grey > 0] = 0
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if not balls:
        return img
    if "spotlight" in modes:
        lit = np.zeros((h, w), np.float32)
        for b in balls:
            cv2.circle(lit, (int(b.x), int(b.y)), int(b.r * 2.0) + 2, 1.0, -1)
        lit = cv2.GaussianBlur(lit, (0, 0), max(2.0, 0.01 * w))[:, :, None]
        dim = (0.45 * cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
               + 0.15 * img).astype(np.float32)
        img = (lit * img + (1 - lit) * dim).clip(0, 255).astype(np.uint8)
    if "box" in modes:
        thick = max(2, round(0.008 * w))
        rgb_ring = RING_BGR[::-1]
        for b in balls:
            cv2.circle(img, (int(b.x), int(b.y)), int(b.r * 1.35) + thick, rgb_ring, thick, cv2.LINE_AA)
    return img


# How each ball colour is drawn (RGB) and said; anything else YOLO calls a ball is drawn yellow, "ball".
_SHOWN = {"orange": ((255, 140, 0), "orange"), "green_ball": ((0, 230, 120), "green"),
          "blue_ball": ((30, 110, 255), "blue"), "yellow_ball": ((220, 230, 40), "yellow"),
          "green_cube": ((150, 210, 90), "green cube"), "cube": ((180, 111, 214), "cube"), "rod": ((224, 80, 63), "rod"),
          "white": ((240, 240, 240), "white"), "black": ((90, 90, 90), "black")}


def ball_color(rgb: npt.NDArray[np.uint8], box: tuple, cal: Calibration, min_frac: float = 0.15) -> str | None:
    """The calibration colour of the ball in a YOLO box: of the match's ball colours (ours, the other team's, the
    never-grab white, the black 8-ball), the one whose HSV range covers most of the box's inner disc. The disc
    (60% of the radius) leaves out the table around the ball. None when no range covers `min_frac` of it."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    disc = np.zeros(crop.shape[:2], np.uint8)
    cv2.circle(disc, ((x1 - x0) // 2, (y1 - y0) // 2), max(1, int(0.3 * min(x1 - x0, y1 - y0))), 255, -1)
    inside = max(1, cv2.countNonZero(disc))
    best, best_frac = None, min_frac
    for color in (*cal.ball_colors, *cal.avoid_balls, cal.eight_ball, *cal.other_balls):
        if color not in cal.colors:
            continue
        frac = cv2.countNonZero(mask_for(crop, cal.colors[color]) & disc) / inside
        if frac > best_frac:
            best, best_frac = color, frac
    return best


def label_balls(rgb: npt.NDArray[np.uint8], boxes: list[tuple[str, float, tuple]], cal: Calibration,
                keep_white: bool = False) -> list[tuple[str | None, float, tuple]]:
    """YOLO's ball boxes with each one's colour (`ball_color`) in place of the class name. White balls (never
    grabbed) and boxes of no ball colour (a reflection, a bottle cap) are left out unless `keep_white`; the black
    8-ball stays, as the one ball that must never reach the bin."""
    named = [(COLOR_CLASSES[n] if n in COLOR_CLASSES else ball_color(rgb, b, cal) if n == "sports ball"
              else n if n in DECOY_CLASSES else None, p, b) for n, p, b in boxes]  # a fine-tuned model's decoys too
    if keep_white:
        return named
    return [n for n in named if n[0] is not None and n[0] not in cal.avoid_balls]


def shown_name(color: str | None) -> str:
    """How a ball colour or decoy class is said in the debug view and the dashboard badge."""
    return _SHOWN.get(color or "", (None, "ball"))[1]


def draw_yolo_boxes(rgb: npt.NDArray[np.uint8], boxes: list[tuple[str | None, float, tuple]]) -> npt.NDArray[np.uint8]:
    """Every YOLO ball box, drawn and labelled in its colour (`label_balls`) with YOLO's confidence."""
    img = rgb.copy()
    scale = max(0.35, img.shape[1] / 1600)
    for name, p, (x0, y0, x1, y1) in boxes:
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        col, word = _SHOWN.get(name or "", ((255, 220, 0), "ball"))
        cv2.rectangle(img, p0, p1, col, 2)
        label = f"{word} {p:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(th + 2, p0[1])
        cv2.rectangle(img, (p0[0], ty - th - 3), (p0[0] + tw + 2, ty), col, -1)
        ink = (255, 255, 255) if word == "black" else (0, 0, 0)
        cv2.putText(img, label, (p0[0] + 1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)
    return img


def decode(jpeg_b64: str) -> npt.NDArray[np.uint8]:
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(jpeg_b64), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode(rgb: npt.NDArray[np.uint8], quality: int = JPEG_QUALITY) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------- the client wrapper
class _Worker:
    """One camera's detector thread: always works on the newest frame handed to it."""

    def __init__(self, key: str, finder: Finder) -> None:
        self.finder = finder
        self._cv = threading.Condition()
        self._job: tuple[npt.NDArray[np.uint8], str] | None = None
        self.result: tuple[float, str, list[Ball]] | None = None  # (when, target, balls)
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=f"augment-{key}", daemon=True)
        self._thread.start()

    def offer(self, rgb: npt.NDArray[np.uint8], target: str) -> None:
        with self._cv:
            self._job = (rgb, target)
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    return
                rgb, target = self._job
                self._job = None
            try:
                balls = self.finder(rgb, target)
            except Exception:  # noqa: BLE001 - a detector fault must never stop the robot: send frames unmarked
                _log.exception("augmentation detector failed")
                balls = []
            self.result = (time.monotonic(), target, balls)


class AugmentingClient:
    """Wraps a Dream client; rewrites the frames (and optionally the prompt) of each `set_observation` call.

    Everything else (get_action, reset_episode, refill_due, close, ...) goes straight to the wrapped client.
    """

    def __init__(self, client: Any, cal: Calibration, modes: tuple[str, ...] = ("box",), detector: str = "both",
                 *, yolo_weights: str = "yolo11n.pt", prompt_note: str | None = None, max_age_s: float = 1.0,
                 sync: bool = False, debug_dir: pathlib.Path | None = None, debug_every_s: float = 2.0) -> None:
        bad = [m for m in modes if m not in MODES]
        if bad:
            raise ValueError(f"augment modes {bad} are not among {MODES}")
        self._client = client
        self.cal, self.modes, self.prompt_note = cal, tuple(modes), prompt_note
        self.max_age_s, self.sync = max_age_s, sync
        self.finder = Finder(cal, detector, yolo_weights)
        self._workers: dict[str, _Worker] = {}
        self._lock = threading.Lock()
        self.debug_dir, self.debug_every_s, self._last_debug = debug_dir, debug_every_s, 0.0
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"frames": 0, "marked": 0, "target_steps": 0}
        self.last_sent: dict[str, tuple[float, str]] = {}  # video key -> (when, base64 JPEG) as sent to Dream
        self._last_target: str | None = None

    def __getattr__(self, name: str) -> Any:  # only called for what this class does not define
        return getattr(self._client, name)

    def set_observation(self, frames: Mapping[str, str], observation: Mapping[str, Any], *,
                        timestamp: float | None = None) -> None:
        try:
            frames, observation = self.augment(frames, observation)
        except Exception:  # noqa: BLE001 - never block the loop: on any fault the frames go out as they came
            _log.exception("augmentation failed; sending the frames unchanged")
        self._client.set_observation(frames, observation, timestamp=timestamp)

    def augment(self, frames: Mapping[str, str], observation: Mapping[str, Any]
                ) -> tuple[Mapping[str, str], Mapping[str, Any]]:
        prompt = str(observation.get("prompt", ""))
        target = target_from_prompt(prompt, self.cal)
        if target != self._last_target:
            _log.info("augmentation target: %s", target or "none (frames pass through)")
            self._last_target = target
            if target:
                self.stats["target_steps"] += 1
        now = time.monotonic()
        if target is None:
            self.last_sent.update({k: (now, v) for k, v in frames.items()})
            return frames, observation
        out: dict[str, str] = {}
        sample: dict[str, npt.NDArray[np.uint8]] = {}
        with self._lock:
            for key, jpeg in frames.items():
                rgb = decode(jpeg)
                balls = self._balls(key, rgb, target, now)
                marked = annotate(rgb, balls, self.modes, target, self.cal)
                out[key] = encode(marked)
                self.last_sent[key] = (now, out[key])
                self.stats["frames"] += 1
                self.stats["marked"] += bool(balls)
                sample[key] = marked
        if self.debug_dir is not None and now - self._last_debug >= self.debug_every_s:
            self._last_debug = now
            strip = np.hstack([cv2.resize(v, (320, 240)) for v in sample.values()])
            cv2.imwrite(str(self.debug_dir / f"sent_{now:.1f}.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        if self.prompt_note and self.prompt_note not in prompt:
            observation = {**observation, "prompt": f"{prompt} {self.prompt_note}"}
        return out, observation

    def _balls(self, key: str, rgb: npt.NDArray[np.uint8], target: str, now: float) -> list[Ball]:
        if self.sync:
            return self.finder(rgb, target)
        worker = self._workers.get(key)
        if worker is None:
            worker = self._workers[key] = _Worker(key, self.finder)
        worker.offer(rgb, target)
        res = worker.result
        if res is None or res[1] != target or now - res[0] > self.max_age_s:
            return []
        return res[2]

    def close(self) -> None:
        for w in self._workers.values():
            w.stop()
        self._workers.clear()  # a later loop on the same client starts fresh workers
        _log.info("augmentation: %s", self.stats)
        self._client.close()


def wrap(client: Any, cal: Calibration, spec: str | None, **kw: Any) -> Any:
    """`client` wrapped per a CLI spec ("off", "box", "box,suppress", ...); the client itself when off."""
    if not spec or spec == "off":
        return client
    return AugmentingClient(client, cal, tuple(m.strip() for m in spec.split(",") if m.strip()), **kw)


# ---------------------------------------------------------------------- offline preview
def main(argv: list[str] | None = None) -> None:
    import argparse

    from harness.config import DEFAULT_CALIBRATION

    ap = argparse.ArgumentParser(prog="harness.augment", description="preview the augmentation on saved frames")
    ap.add_argument("frames", nargs="+", type=pathlib.Path, help="images, or .mp4 videos (every --every-th frame)")
    ap.add_argument("--target", default="green_ball")
    ap.add_argument("--mode", default="box,suppress")
    ap.add_argument("--detector", default="both", choices=DETECTORS)
    ap.add_argument("--yolo-weights", default="yolo11n.pt")
    ap.add_argument("--every", type=int, default=60, help="for videos: keep one frame in this many")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="output folder (default: next to each input)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cal = Calibration.load(DEFAULT_CALIBRATION)
    modes = tuple(args.mode.split(","))
    finder = Finder(cal, args.detector, args.yolo_weights)

    def frames_of(path: pathlib.Path):
        if path.suffix.lower() in (".mp4", ".mov", ".avi"):
            cap = cv2.VideoCapture(str(path))
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % args.every == 0:
                    yield f"{path.stem}_{i:05d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                i += 1
        else:
            yield path.stem, cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    for path in args.frames:
        out_dir = args.out or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, rgb in frames_of(path):
            t = time.perf_counter()
            balls = finder(rgb, args.target)
            ms = (time.perf_counter() - t) * 1000
            marked = annotate(rgb, balls, modes, args.target, cal)
            dst = out_dir / f"{name}_aug.jpg"
            cv2.imwrite(str(dst), cv2.cvtColor(np.hstack([rgb, marked]), cv2.COLOR_RGB2BGR))
            found = ", ".join(f"{b.source} ({b.x:.0f},{b.y:.0f}) r{b.r:.0f}" for b in balls) or "none"
            print(f"{name}: {len(balls)} found in {ms:.0f} ms [{found}] -> {dst}")


if __name__ == "__main__":
    main()
