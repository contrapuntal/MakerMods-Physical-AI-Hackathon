"""The augmenting client on a synthetic frame: a green ball and a green cube, in the calibration's own colours."""

import time

import cv2
import numpy as np

from harness import augment
from harness.config import DEFAULT_CALIBRATION, Calibration

CAL = Calibration.load(DEFAULT_CALIBRATION)
BALL_PROMPT = "pick up the green ping-pong ball and drop it in the bin and do not touch the white or the black balls"


def _hsv_px(h: int, s: int, v: int) -> tuple[int, int, int]:
    rgb = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(c) for c in rgb)


def _frame() -> np.ndarray:
    img = np.full((240, 320, 3), 200, np.uint8)
    cv2.circle(img, (80, 120), 12, _hsv_px(84, 200, 200), -1)  # green_ball: H76-92, S>=140
    cv2.rectangle(img, (200, 100), (232, 132), _hsv_px(65, 100, 180), -1)  # green cube: H52-75, S45-150
    return img


class FakeClient:
    def __init__(self) -> None:
        self.calls = []
        self.session_id = "abc"
        self.closed = False

    def set_observation(self, frames, observation, *, timestamp=None):
        self.calls.append((dict(frames), dict(observation), timestamp))

    def close(self):
        self.closed = True


def test_target_from_prompt():
    assert augment.target_from_prompt(BALL_PROMPT, CAL) == "green_ball"
    assert augment.target_from_prompt("pick up the orange ping-pong ball and drop it", CAL) == "orange"
    assert augment.target_from_prompt("put the green cube on the purple cube", CAL) is None
    assert augment.target_from_prompt("take the green ping-pong ball out of the bin", CAL) is None


def test_hsv_finds_the_ball_not_the_cube():
    balls = augment.find_hsv(_frame(), "green_ball", CAL)
    assert len(balls) == 1 and abs(balls[0].x - 80) < 2 and abs(balls[0].y - 120) < 2


def test_box_and_suppress():
    img = _frame()
    out = augment.annotate(img, augment.find_hsv(img, "green_ball", CAL), ("box", "suppress"), "green_ball", CAL)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    assert hsv[116, 216, 1] < 10  # the cube is grey now
    assert hsv[120, 80, 1] > 140  # the ball keeps its colour
    ring = out[120, 80 + 12 + 6]  # on the ring, just outside the ball
    assert ring[0] > 200 and ring[2] > 200 and ring[1] < 80  # magenta


def test_client_marks_ball_steps_and_passes_the_rest():
    fake = FakeClient()
    client = augment.AugmentingClient(fake, CAL, ("box",), detector="hsv", sync=True, prompt_note="(circled)")
    jpeg = augment.encode(_frame())
    client.set_observation({"video.overhead": jpeg}, {"prompt": "put the green cube on the purple cube"},
                           timestamp=1.0)
    assert fake.calls[-1][0]["video.overhead"] is jpeg  # untouched off a ball step
    client.set_observation({"video.overhead": jpeg}, {"prompt": BALL_PROMPT, "state.x": 1}, timestamp=2.0)
    frames, obs, ts = fake.calls[-1]
    assert frames["video.overhead"] != jpeg and ts == 2.0
    assert obs["prompt"].endswith("(circled)") and obs["state.x"] == 1
    assert client.session_id == "abc"  # everything else goes to the wrapped client
    client.close()
    assert fake.closed


def test_async_worker_catches_up():
    fake = FakeClient()
    client = augment.AugmentingClient(fake, CAL, ("box",), detector="hsv")
    jpeg = augment.encode(_frame())
    client.set_observation({"video.overhead": jpeg}, {"prompt": BALL_PROMPT})
    end = time.monotonic() + 2
    while client.stats["marked"] == 0 and time.monotonic() < end:
        time.sleep(0.02)
        client.set_observation({"video.overhead": jpeg}, {"prompt": BALL_PROMPT})
    assert client.stats["marked"] >= 1
    client.close()


def test_a_detector_fault_sends_frames_unchanged():
    fake = FakeClient()
    client = augment.AugmentingClient(fake, CAL, ("box",), detector="hsv", sync=True)
    client.finder = lambda rgb, target: 1 / 0
    jpeg = augment.encode(_frame())
    client.set_observation({"video.overhead": jpeg}, {"prompt": BALL_PROMPT})
    assert fake.calls[-1][0]["video.overhead"] is jpeg


def test_ball_color_names_each_ball():
    img = _frame()
    cv2.circle(img, (150, 60), 12, _hsv_px(8, 150, 220), -1)  # orange: H3-14, S90-210
    cv2.circle(img, (150, 190), 12, (10, 10, 10), -1)  # black 8-ball
    assert augment.ball_color(img, (68, 108, 92, 132), CAL) == "green_ball"
    assert augment.ball_color(img, (138, 48, 162, 72), CAL) == "orange"
    assert augment.ball_color(img, (138, 178, 162, 202), CAL) == "black"
    assert augment.ball_color(img, (10, 10, 34, 34), CAL) is None  # bare table


def test_label_balls_drops_white_and_unnamed():
    img = _frame()
    cv2.circle(img, (150, 60), 12, _hsv_px(107, 42, 184), -1)  # a white ball (calibration's sample)
    boxes = [("sports ball", 0.5, (68, 108, 92, 132)), ("sports ball", 0.4, (138, 48, 162, 72)),
             ("sports ball", 0.3, (10, 10, 34, 34))]
    assert [c for c, _, _ in augment.label_balls(img, boxes, CAL)] == ["green_ball"]
    assert [c for c, _, _ in augment.label_balls(img, boxes, CAL, keep_white=True)] == ["green_ball", "white", None]
