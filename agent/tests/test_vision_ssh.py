"""Station camera over ssh: only the newest frame of one camera is picked on the station and sent back."""

import asyncio
import base64
import subprocess
import sys
import time

import pytest

import vision

JPEG = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"


def _reply(n=7, key="video.scene", jpeg=JPEG):
    return 0, f"{n} {key} {base64.b64encode(jpeg).decode()}", ""


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    vision._seen.clear()
    monkeypatch.setenv("CAMERA_STATION_SSH", "nuck-3")
    yield
    vision._seen.clear()


def _fake(monkeypatch, replies):
    calls = []

    async def run(argv, script):
        calls.append(argv)
        r = replies.pop(0) if replies else _reply()
        if isinstance(r, BaseException):
            raise r
        if r == "hang":
            await asyncio.sleep(10)
        return r

    monkeypatch.setattr(vision, "_run_ssh", run)
    return calls


def test_argv_uses_the_helper_master_and_asks_for_the_role_keys():
    argv = vision._ssh_frame_argv("nuck-3", vision.STATION_KEYS["overhead"])
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and any("pah2026-helper-%h.sock" in a for a in argv)
    i = argv.index("--")
    assert argv[i + 1 :] == ["nuck-3", "python3", "-", "http://127.0.0.1:8093", "video.overhead", "video.scene"]


def test_bad_host_is_refused():
    with pytest.raises(vision.CameraDown):
        vision._ssh_frame_argv("nuck-3; rm -rf ~", ("video.scene",))


def test_fetch_station_routes_to_ssh_and_returns_the_jpeg(monkeypatch):
    calls = _fake(monkeypatch, [_reply()])
    assert asyncio.run(vision.fetch_station("overhead")) == JPEG
    assert len(calls) == 1 and calls[0][-2:] == ["video.overhead", "video.scene"]


def test_wrist_role_keys(monkeypatch):
    calls = _fake(monkeypatch, [_reply(key="video.wrist_l")])
    asyncio.run(vision.fetch_station("wrist_l"))
    assert calls[0][-1] == "video.wrist_l"


def test_station_errors_become_camera_down(monkeypatch):
    _fake(monkeypatch, [(4, "", "station view has no round trips yet")])
    with pytest.raises(vision.CameraDown, match="no round trips"):
        asyncio.run(vision.fetch_station("overhead"))
    _fake(monkeypatch, [(0, "garbage", "")])
    with pytest.raises(vision.CameraDown, match="malformed"):
        asyncio.run(vision.fetch_station("overhead"))
    _fake(monkeypatch, [OSError("no ssh")])
    with pytest.raises(vision.CameraDown):
        asyncio.run(vision.fetch_station("overhead"))


def test_timeout_is_camera_down(monkeypatch):
    monkeypatch.setattr(vision, "STATION_SSH_TIMEOUT_S", 0.1)
    _fake(monkeypatch, ["hang"])
    with pytest.raises(vision.CameraDown):
        asyncio.run(vision.fetch_station("overhead"))


def test_same_trip_for_too_long_is_stale(monkeypatch):
    monkeypatch.setenv("CAMERA_STATION_STALE_S", "0.2")
    _fake(monkeypatch, [_reply(n=5), _reply(n=5), _reply(n=6)])
    assert asyncio.run(vision.fetch_station("overhead")) == JPEG
    time.sleep(0.3)
    with pytest.raises(vision.CameraDown, match="only update while a Dream run is active"):
        asyncio.run(vision.fetch_station("overhead"))
    assert vision._station_stale
    assert asyncio.run(vision.fetch_station("overhead")) == JPEG  # a new trip number: fresh again
    assert not vision._station_stale


def test_unset_host_keeps_the_http_path(monkeypatch):
    monkeypatch.delenv("CAMERA_STATION_SSH")
    called = _fake(monkeypatch, [])
    assert vision.station_ssh_host() is None
    with pytest.raises(vision.CameraDown):  # no view on this port in tests: the HTTP path, not ssh
        asyncio.run(vision.fetch_station("overhead", base_url="http://127.0.0.1:9"))
    assert called == []


def test_station_script_against_a_real_debugview():
    """The script the station runs, against Dream's real DebugView on this machine (no ssh)."""
    dream = pytest.importorskip("dream")
    from dream.debug_view import RoundTrip

    view = dream.DebugView(None, port=0)
    try:
        b64 = base64.b64encode(JPEG).decode()
        view._add(RoundTrip(sent_s=time.time(), frames={"video.scene": ["old", b64], "video.wrist_l": [b64]},
                            observation={}, action=None, timing=None, error=None))
        base = view.url.rstrip("/")
        r = subprocess.run([sys.executable, "-", base, "video.overhead", "video.scene"], input=vision._SSH_FRAME_SCRIPT,
                           capture_output=True, text=True, timeout=20)
        assert r.returncode == 0, r.stderr
        n, key, frame = r.stdout.split(" ", 2)
        assert key == "video.scene" and base64.b64decode(frame) == JPEG  # the newest frame of the window
        r = subprocess.run([sys.executable, "-", base, "video.wrist_r"], input=vision._SSH_FRAME_SCRIPT,
                           capture_output=True, text=True, timeout=20)
        assert r.returncode == 5 and "no frame" in r.stderr
    finally:
        view.close()
    r = subprocess.run([sys.executable, "-", "http://127.0.0.1:9", "video.scene"], input=vision._SSH_FRAME_SCRIPT,
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 3 and "unreachable" in r.stderr
