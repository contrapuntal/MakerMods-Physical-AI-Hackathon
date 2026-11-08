"""Publish the station's camera snapshots as privacy-processed LiveKit video tracks (view-only).

Run: ( set -a; . ~/.config/pah2026/secrets.sh; set +a; uv run python src/camera_relay.py )
Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, CAMERA_ROOM (default pah2026-cameras),
CAMERA_FPS (default 2), CAMERA_SOURCE_URL, CAMERA_RELAY_WRISTS=1 (opt in; wrists are not relayed by default),
CAMERA_BAND_OVERHEAD / CAMERA_BAND_WRIST (privacy band fractions).
Every frame goes through vision.process (band, then YuNet face scrub, fail closed). Never talks to the station.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

import cv2
from livekit import api, rtc

import vision

logger = logging.getLogger("camera_relay")

IDENTITY = "camera-relay"


def make_token(room: str) -> str:
    # AccessToken reads LIVEKIT_API_KEY / LIVEKIT_API_SECRET from the environment.
    return (
        api.AccessToken()
        .with_identity(IDENTITY)
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=False))
        .to_jwt()
    )


def relay_roles() -> tuple[str, ...]:
    return vision.ROLES if os.environ.get("CAMERA_RELAY_WRISTS") == "1" else ("overhead",)


async def relay_camera(
    role: str, source: rtc.VideoSource, fps: float, stop: asyncio.Event, session=None
) -> None:
    """Poll one camera and push safe frames. Logs once per outage; sends NO SIGNAL on any failure."""
    interval = 1.0 / max(fps, 0.1)
    loop = asyncio.get_running_loop()
    down = False
    while not stop.is_set():
        t0 = loop.time()
        try:
            jpeg = await vision.fetch_jpeg(role, session=session)
            frame = await asyncio.to_thread(vision.frame_for_publish, role, jpeg)
            if down:
                logger.info("%s: signal restored", role)
            down = False
        except Exception as exc:  # noqa: BLE001 - a track must never freeze silently
            if not down:
                logger.warning("%s: no signal (%r); retrying every %.1fs", role, exc, interval)
            down = True
            frame = vision.to_video_frame(vision.no_signal_frame())
        source.capture_frame(frame)
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval - (loop.time() - t0)))
        except TimeoutError:
            pass


async def main() -> int:
    logging.basicConfig(level=logging.INFO)
    cv2.setNumThreads(2)
    url = os.environ["LIVEKIT_URL"]
    room_name = os.environ.get("CAMERA_ROOM") or "pah2026-cameras"
    fps = float(os.environ.get("CAMERA_FPS") or 2)

    await vision.warmup()  # off-loop; failure just means NO SIGNAL frames (fail closed)

    stop = asyncio.Event()
    exit_code = 0
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    room = rtc.Room()

    @room.on("disconnected")
    def _on_disconnected(reason=None) -> None:
        nonlocal exit_code
        if not stop.is_set():
            logger.error("disconnected from the room (%s); exiting", reason)
            exit_code = 1
            stop.set()

    await room.connect(url, make_token(room_name))
    logger.info("connected to room %s as %s", room_name, IDENTITY)
    session = vision.get_session()
    tasks = []
    try:
        for role in relay_roles():
            source = rtc.VideoSource(640, 360)
            track = rtc.LocalVideoTrack.create_video_track(role, source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            )
            tasks.append(asyncio.create_task(relay_camera(role, source, fps, stop, session)))
        await stop.wait()
    finally:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await vision.close_session()
        await room.disconnect()
        logger.info("disconnected")
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
