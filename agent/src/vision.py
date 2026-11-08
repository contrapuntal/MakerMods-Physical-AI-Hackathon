"""View-only camera helpers shared by the relay and the agent's vision tools.

Fetches JPEG snapshots from the kit's dashboard (CAMERA_SOURCE_URL). EVERY exit (relay frames, images to the
vision LLM, anything logged or saved) goes through `process()`: privacy band -> YuNet face scrub -> crop/scale.
The band and the YuNet scrub are always on and fail closed (no env var removes them).
Nothing here talks to the robot station.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time

import aiohttp
import cv2
import numpy as np
from shared.faceblur import FaceBlurError, blurrer_for

logger = logging.getLogger("vision")

DEFAULT_SOURCE_URL = "http://127.0.0.1:8080"
ROLES = ("overhead", "wrist_l", "wrist_r")
WRIST_ROLES = ("wrist_l", "wrist_r")
FETCH_TIMEOUT_S = 3.0
# Top fraction of the frame that is always filled (people sit behind the table).
DEFAULT_BAND = {"overhead": 0.42, "wrist_l": 0.50, "wrist_r": 0.50}
BAND_FILL = (40, 40, 40)


class CameraDown(Exception):
    """The camera source is unreachable, returned no usable image, or the privacy scrub is unavailable."""


def source_url() -> str:
    return (os.environ.get("CAMERA_SOURCE_URL") or DEFAULT_SOURCE_URL).rstrip("/")


def band_fraction(role: str) -> float:
    name = "CAMERA_BAND_OVERHEAD" if role == "overhead" else "CAMERA_BAND_WRIST"
    try:
        v = float(os.environ.get(name, ""))
    except ValueError:
        return DEFAULT_BAND[role]
    return min(1.0, max(0.0, v))


# One shared aiohttp session per event loop.
_session: aiohttp.ClientSession | None = None
_session_loop: asyncio.AbstractEventLoop | None = None


def get_session() -> aiohttp.ClientSession:
    global _session, _session_loop
    loop = asyncio.get_running_loop()
    if _session is None or _session.closed or _session_loop is not loop:
        _session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_S))
        _session_loop = loop
    return _session


async def close_session() -> None:
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


DEFAULT_STATION_URL = "http://127.0.0.1:8093"
STATION_TRIP_TTL_S = 0.5
_trip_cache: tuple[float, int, dict, str] | None = None
_last_ok: str | None = None  # "dashboard" | "station": tried first in auto mode


def station_url() -> str:
    return (os.environ.get("CAMERA_STATION_URL") or DEFAULT_STATION_URL).rstrip("/")


def source_mode() -> str:
    m = (os.environ.get("CAMERA_SOURCE") or "auto").strip().lower()
    return m if m in ("auto", "dashboard", "station") else "auto"


def down_message() -> str:
    """One spoken line naming the program that has to be running."""
    mode = source_mode()
    if mode == "dashboard":
        return "I can't see the cameras. The kit's dhk-collect program needs to be running."
    if mode == "station":
        if _station_stale:
            return "I can't see the cameras. The Dream station cameras only update while a Dream run is active, so start a run."
        return "I can't see the cameras. The Dream station agent needs to be running."
    if _station_stale:
        return (
            "I can't see the cameras. Start the kit's dhk-collect program, "
            "or start a Dream run, since the station cameras only update during a run."
        )
    return "I can't see the cameras. Either the kit's dhk-collect program or the Dream station agent needs to be running."


async def fetch_dashboard(role: str, base_url: str | None = None, session=None) -> bytes:
    """dhk-collect dashboard: GET <url>/api/camera/<role>.jpg (a plain JPEG)."""
    url = f"{(base_url or source_url()).rstrip('/')}/api/camera/{role}.jpg"
    try:
        async with (session or get_session()).get(url) as resp:
            if resp.status != 200:
                raise CameraDown(f"camera source answered HTTP {resp.status} for {role}")
            return await resp.read()
    except (aiohttp.ClientError, TimeoutError, OSError) as exc:
        raise CameraDown(f"camera source unreachable: {exc!r}") from exc


# Dream station view (dream/debug_view.py): GET /trips -> [{"n", "sent_s", "error", "rtt_ms"}, ...] oldest first;
# GET /trip/<n> -> {"frames": {"video.<role>": [base64 JPEG, ... oldest first], ...}, ...}. Roles seen in the
# source: scene (the overhead camera), overhead, wrist_l, wrist_r. Derived from the source, not a live page.
STATION_KEYS = {
    "overhead": ("video.overhead", "video.scene"),
    "wrist_l": ("video.wrist_l",),
    "wrist_r": ("video.wrist_r",),
}


def parse_station_trip(trip: dict, role: str) -> bytes:
    """Newest JPEG for `role` out of a /trip/<n> payload. Raises CameraDown if absent or malformed."""
    frames = trip.get("frames") if isinstance(trip, dict) else None
    if not isinstance(frames, dict):
        raise CameraDown("station view reply has no frames")
    for key in STATION_KEYS[role]:
        lst = frames.get(key)
        if isinstance(lst, str):
            lst = [lst]
        if lst:
            try:
                return base64.b64decode(lst[-1], validate=True)
            except ValueError as exc:
                raise CameraDown("station view frame is not base64") from exc
    raise CameraDown(f"station view has no {role} frame")


STATION_STALE_MSG = "the station cameras only update while a Dream run is active"
_seen: dict[
    str, tuple[int, float]
] = {}  # base url -> (newest trip n, laptop monotonic time it was first seen)
_station_stale = False
_trip_lock: tuple[asyncio.AbstractEventLoop, asyncio.Lock] | None = None


def stale_after_s() -> float:
    try:
        return float(os.environ.get("CAMERA_STATION_STALE_S", ""))
    except ValueError:
        return 3.0


def _lock() -> asyncio.Lock:
    global _trip_lock
    loop = asyncio.get_running_loop()
    if _trip_lock is None or _trip_lock[0] is not loop:
        _trip_lock = (loop, asyncio.Lock())
    return _trip_lock[1]


async def _refresh_trip(base: str, sess) -> None:
    """Fetch the newest trip once for everyone (caller holds the lock). Updates _trip_cache and _seen."""
    global _trip_cache
    try:
        async with sess.get(f"{base}/trips") as resp:
            if resp.status != 200:
                raise CameraDown(f"station view answered HTTP {resp.status}")
            trips = await asyncio.to_thread(json.loads, await resp.read())
        if (
            not isinstance(trips, list)
            or not trips
            or not isinstance(trips[-1], dict)
            or "n" not in trips[-1]
        ):
            raise CameraDown("station view has no round trips yet")
        n = int(trips[-1]["n"])
        async with sess.get(f"{base}/trip/{n}") as resp:
            if resp.status != 200:
                raise CameraDown(f"station view answered HTTP {resp.status} for trip {n}")
            trip = await asyncio.to_thread(json.loads, await resp.read())
    except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as exc:
        raise CameraDown(f"station view unreachable or malformed: {exc!r}") from exc
    now = time.monotonic()
    if base not in _seen or _seen[base][0] != n:
        _seen[base] = (n, now)  # new trip number: restart the staleness clock (laptop clock only)
    _trip_cache = (now, n, trip, base)


async def fetch_station(role: str, base_url: str | None = None, session=None) -> bytes:
    """Newest station frame for a role. Concurrent callers share one refresh (same trip for all roles).

    Raises CameraDown if the newest trip number has not changed for CAMERA_STATION_STALE_S (default 3 s): the
    debug view keeps old trips and only adds new ones while a Dream run is active.
    """
    global _station_stale
    base = (base_url or station_url()).rstrip("/")
    async with _lock():
        c = _trip_cache
        if c is None or c[3] != base or time.monotonic() - c[0] >= STATION_TRIP_TTL_S:
            await _refresh_trip(base, session or get_session())
        trip = _trip_cache[2]
        _, first_seen = _seen[base]
        if time.monotonic() - first_seen >= stale_after_s():
            _station_stale = True
            raise CameraDown(STATION_STALE_MSG)
        _station_stale = False
    return parse_station_trip(trip, role)


async def fetch_jpeg(role: str, base_url: str | None = None, session=None) -> bytes:
    """Latest raw JPEG for a role from the configured source (CAMERA_SOURCE: auto|dashboard|station).

    An explicit base_url means the dashboard only. Auto tries the source that last worked first.
    """
    global _last_ok
    if role not in ROLES:
        raise ValueError(f"unknown camera {role!r}")
    if base_url:
        return await fetch_dashboard(role, base_url, session)
    mode = source_mode()
    if mode == "dashboard":
        return await fetch_dashboard(role, None, session)
    if mode == "station":
        return await fetch_station(role, None, session)
    order = ["dashboard", "station"] if _last_ok != "station" else ["station", "dashboard"]
    errors = []
    for name in order:
        try:
            jpeg = await (fetch_dashboard if name == "dashboard" else fetch_station)(role, None, session)
        except CameraDown as exc:
            errors.append(f"{name}: {exc}")
            continue
        _last_ok = name
        return jpeg
    raise CameraDown("; ".join(errors))


def decode(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise CameraDown("camera source returned an undecodable image")
    return img


def encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CameraDown("could not encode frame")
    return buf.tobytes()


def apply_band(img: np.ndarray, role: str) -> np.ndarray:
    """Solid-fill the top fraction of the frame (copy)."""
    out = img.copy()
    rows = round(out.shape[0] * band_fraction(role))
    out[:rows] = BAND_FILL
    return out


def scrub_faces(img: np.ndarray, role: str) -> np.ndarray:
    """YuNet scrub via Dream Labs' shared.faceblur (fails closed). Raises CameraDown."""
    try:
        return blurrer_for(role).scrub_frame(img)
    except FaceBlurError as exc:
        raise CameraDown(f"face blur unavailable: {exc}") from exc


def process(img: np.ndarray, role: str, scale: float | None = None, crop=None) -> np.ndarray:
    """band -> YuNet scrub -> crop/scale. Raises CameraDown if the scrub cannot run."""
    img = scrub_faces(apply_band(img, role), role)
    if crop is not None:
        x, y, w, h = crop
        img = img[y : y + h, x : x + w]
    if scale and scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def process_jpeg_to_frame(role: str, jpeg: bytes) -> np.ndarray:
    """Sync (run in a thread): decode -> process. Returns the safe BGR frame."""
    return process(decode(jpeg), role)


def process_jpeg(role: str, jpeg: bytes) -> bytes:
    """Sync (run in a thread): decode -> process -> encode."""
    return encode_jpeg(process_jpeg_to_frame(role, jpeg))


async def get_blurred_jpeg(role: str, base_url: str | None = None) -> bytes:
    """Latest snapshot as a safe JPEG (banded and face-scrubbed). Raises CameraDown."""
    jpeg = await fetch_jpeg(role, base_url)
    return await asyncio.to_thread(process_jpeg, role, jpeg)


def warmup_sync() -> None:
    for role in ROLES:
        blurrer_for(role)  # constructs and self-checks the detector


async def warmup() -> bool:
    """Load the blurrers off the event loop so the first look does not stall. False if unavailable."""
    try:
        await asyncio.to_thread(warmup_sync)
        return True
    except FaceBlurError as exc:
        logger.error("face blur unavailable, camera features will report no signal: %s", exc)
        return False


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def no_signal_frame(width: int = 640, height: int = 360) -> np.ndarray:
    img = np.full((height, width, 3), 40, np.uint8)
    cv2.putText(
        img, "NO SIGNAL", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2
    )
    return img


def to_video_frame(img: np.ndarray):
    """BGR ndarray -> livekit rtc.VideoFrame (RGBA)."""
    from livekit import rtc

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return rtc.VideoFrame(rgba.shape[1], rgba.shape[0], rtc.VideoBufferType.RGBA, rgba.tobytes())


def frame_for_publish(role: str, jpeg: bytes):
    """Sync (run in a thread): decode -> process -> VideoFrame."""
    return to_video_frame(process_jpeg_to_frame(role, jpeg))
