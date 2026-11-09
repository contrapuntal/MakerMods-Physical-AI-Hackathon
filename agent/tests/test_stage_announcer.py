"""Stage announcements from the station view's current instruction (no robot, no ssh: fakes)."""

import asyncio

import pytest

import stage_announcer as sa

# Every instruction the harness sends (harness/relay.py texts and the per-stage fine-tune texts).
CASES = [
    ("Remove Orange Cables from their port", "unplug"),
    ("grab the orange USB cable's plug at the power strip, then pull it straight up out of the port without touching "
     "the blocks or the balls", "unplug"),
    ("pinch the orange plug where it goes into the power strip and lift it straight up until it comes out without "
     "touching the blocks or the balls", "unplug"),
    ("pull one orange USB plug straight up out of the power strip without touching the blocks or the balls", "unplug"),
    ("Stack the green, purple, and blue blocks", "stack"),
    ("put the red rod upright in the hole of the green cube without touching the balls or the cables", "stack"),
    ("place the blue cube on top of the purple cube without touching the balls or the cables", "stack"),
    ("pick up the blue cube with the red rod standing in it by the sides of the cube and place it on top of the purple "
     "cube without touching the balls or the cables", "stack"),
    ("put the red rod upright in the hole of the blue cube on top of the tower without touching the balls or the cables",
     "stack"),
    ("move both arms up and away from the tower and hold them still", None),
    ("Place yellow, blue, green, and orange balls into basket.", "balls"),
    ("Place all yellow, orange, green, and blue balls in the basket. Push away all white balls.", "balls"),
    ("pick up the green ping-pong ball and drop it in the bin and do not touch the white or the black balls", "balls"),
    ("take the white ping-pong ball out of the bin and put it on the table", "balls"),
    ("take the black ball out of the bin and put it on the table", "eight_ball"),
    ("put the orange cable down on the empty table beside the power strip, away from the blocks and the balls, and open "
     "the gripper", None),
    ("", None),
    (None, None),
]


@pytest.mark.parametrize("prompt,stage", CASES)
def test_classify(prompt, stage):
    assert sa.classify(prompt) == stage


def test_lines_have_no_emergency_words():
    from robot_gate import FREEZE_WORD_RE

    assert not any(FREEZE_WORD_RE.search(line) for line in sa.LINES.values())
    assert sa.LINES["unplug"] == "Picking up the cables now."


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _announcer(replies):
    said, clock = [], Clock()
    a = sa.StageAnnouncer("nuck-3", said.append, clock=clock)

    async def fetch():
        r = replies.pop(0)
        return r

    a.fetch_prompt = fetch
    return a, said, clock


def test_announces_on_stage_change_only_and_rearms_after_the_run():
    a, said, clock = _announcer([
        None,                                                   # no run yet
        (1, "Remove Orange Cables from their port"),            # run starts
        (2, "Remove Orange Cables from their port"),            # same stage: quiet
        (5, "move both arms up and away from the tower and hold them still"),  # hold: quiet
        (9, "Stack the green, purple, and blue blocks"),
        None, None,                                             # the view goes away
        (1, "Stack the green, purple, and blue blocks"),        # next run (practice from stack): announced again
    ])
    times = [0, 2, 4, 6, 8, 10, 20, 22]
    for t in times:
        clock.t = t
        asyncio.run(a.step())
    assert said == ["Picking up the cables now.", "Stacking the blocks now.", "Stacking the blocks now."]


def test_a_say_error_does_not_stop_the_announcer():
    def boom(line):
        raise RuntimeError("closed")

    a = sa.StageAnnouncer("nuck-3", boom, clock=Clock())

    async def fetch():
        return 1, "Stack the green, purple, and blue blocks"

    a.fetch_prompt = fetch
    assert asyncio.run(a.step()) == "Stacking the blocks now."


def test_enabled_switch():
    assert sa.enabled({}) and sa.enabled({"STAGE_ANNOUNCE": "1"})
    assert not sa.enabled({"STAGE_ANNOUNCE": "0"}) and not sa.enabled({"STAGE_ANNOUNCE": "off"})


def test_prompt_script_against_a_real_debugview():
    """The script the station runs, against Dream's real DebugView on this machine (no ssh)."""
    import subprocess
    import sys
    import time

    dream = pytest.importorskip("dream")
    from dream.debug_view import RoundTrip

    view = dream.DebugView(None, port=0)
    try:
        view._add(RoundTrip(sent_s=time.time(), frames={"video.scene": ["x"]},
                            observation={"prompt": "Stack the green, purple, and blue blocks"},
                            action=None, timing=None, error=None))
        r = subprocess.run([sys.executable, "-", view.url.rstrip("/")], input=sa._PROMPT_SCRIPT, capture_output=True,
                           text=True, timeout=20)
        assert r.returncode == 0, r.stderr
        assert r.stdout == "1\tStack the green, purple, and blue blocks"
    finally:
        view.close()


def test_unchanged_trip_skips_the_full_read(monkeypatch):
    import vision

    calls = []

    async def run(argv, script):
        calls.append(argv[-1])
        if argv[-1].isdigit():  # the laptop passed the last trip number: the station answers "unchanged"
            return 0, f"{argv[-1]}\t=", ""
        return 0, "7\tStack the green, purple, and blue blocks", ""

    monkeypatch.setattr(vision, "_run_ssh", run)
    a = sa.StageAnnouncer("nuck-3", lambda line: None)
    assert asyncio.run(a.fetch_prompt()) == (7, "Stack the green, purple, and blue blocks")
    assert asyncio.run(a.fetch_prompt()) == (7, "Stack the green, purple, and blue blocks")  # from the cache
    assert calls == ["http://127.0.0.1:8093", "7"]


def test_station_script_says_unchanged_for_the_same_trip():
    import subprocess
    import sys
    import time

    dream = pytest.importorskip("dream")
    from dream.debug_view import RoundTrip

    view = dream.DebugView(None, port=0)
    try:
        view._add(RoundTrip(sent_s=time.time(), frames={}, observation={"prompt": "Remove Orange Cables from their port"},
                            action=None, timing=None, error=None))
        base = view.url.rstrip("/")
        r = subprocess.run([sys.executable, "-", base, "1"], input=sa._PROMPT_SCRIPT, capture_output=True, text=True,
                           timeout=20)
        assert r.returncode == 0 and r.stdout == "1\t="
    finally:
        view.close()
