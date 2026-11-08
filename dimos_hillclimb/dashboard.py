"""The match dashboard: one web page beside the HillClimbStation module, for the sim and the real station alike.

    http://127.0.0.1:8770/          the page (dashboard.html)
    GET  /api/state                 referee summary, tower, bin, the skill-call log, backend
    GET  /api/frame.jpg             the overhead frame (on the station: with the server's detections drawn on)
    POST /api/start  {own_color, seed}   a new match (as the start_match skill)
    POST /api/park                  both arms to rest / park
    POST /api/agent  {message}      text to DimOS's agent (through the MCP server's agent_send), e.g. the go

It lives in the skills module because that is where the referee (stage, score, clock) and the backend both are; the
station server alone doesn't know the score. Standard library only.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

_log = logging.getLogger(__name__)

PAGE = pathlib.Path(__file__).with_name("dashboard.html")
FRAME_MAX_AGE_S = 0.25  # frames are rendered/fetched at most this often, however many viewers poll


class CallLog:
    """Every skill call, newest last: what was asked, what came back, how long it took."""

    def __init__(self, size: int = 300) -> None:
        self.items: deque[dict] = deque(maxlen=size)
        self.lock = threading.Lock()
        self.seq = 0
        self.running: dict | None = None

    def begin(self, skill: str, args: dict) -> dict:
        with self.lock:
            self.seq += 1
            entry = {"id": self.seq, "skill": skill, "args": args, "t": time.time(), "status": "running"}
            self.running = entry
            return entry

    def end(self, entry: dict, status: str, message: str, match_s: float | None) -> None:
        with self.lock:
            entry.update(status=status, message=message, seconds=round(time.time() - entry["t"], 1),
                         match_s=None if match_s is None else round(match_s, 1))
            self.items.append(entry)
            if self.running is entry:
                self.running = None

    def clear(self) -> None:
        with self.lock:
            self.items.clear()

    def snapshot(self) -> dict:
        with self.lock:
            return {"calls": list(self.items), "running": dict(self.running) if self.running else None}


class Dashboard:
    def __init__(self, station_getter: Callable[[], Any], log: CallLog, backend: str, host: str, port: int,
                 mcp_url: str = "http://127.0.0.1:9990/mcp") -> None:
        self.get_station, self.log, self.backend, self.mcp_url = station_getter, log, backend, mcp_url
        self._frame: tuple[float, bytes] | None = None
        self._frame_lock = threading.Lock()
        self._scene: tuple[float, dict] | None = None
        self.http = ThreadingHTTPServer((host, port), self._handler())
        self.url = f"http://{host}:{port}/"

    def start(self) -> None:
        threading.Thread(target=self.http.serve_forever, name="hillclimb-dashboard", daemon=True).start()
        _log.info("dashboard on %s", self.url)

    def stop(self) -> None:
        self.http.shutdown()
        self.http.server_close()

    # ------------------------------------------------------------------ data
    def frame_jpeg(self) -> bytes:
        import cv2

        with self._frame_lock:
            if self._frame and time.monotonic() - self._frame[0] < FRAME_MAX_AGE_S:
                return self._frame[1]
            station = self.get_station()
            rgb = self._nowait(station, "overhead_rgb")
            if rgb is None:  # a sim motion holds the scene: show the frame it rendered last
                rgb = getattr(station, "latest_rgb", None)
                if rgb is None:
                    rgb = np.zeros((360, 640, 3), np.uint8)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._frame = (time.monotonic(), jpg.tobytes())
            return self._frame[1]

    @staticmethod
    def _nowait(station: Any, method: str):
        """Calls a sim method only if the sim's lock is free (the station backend has no lock: it asks the server)."""
        lock = getattr(station, "lock", None)
        if lock is None:
            return getattr(station, method)()
        if not lock.acquire(timeout=0.05):
            return None
        try:
            return getattr(station, method)()
        finally:
            lock.release()

    def state(self) -> dict:
        station = self.get_station()
        rules = station.rules
        scene = self._nowait(station, "scene") if (not self._scene or time.monotonic() - self._scene[0] > 0.4) else None
        if scene is not None:
            self._scene = (time.monotonic(), scene)
        scene = self._scene[1] if self._scene else {}
        return {
            "backend": self.backend,
            "match": rules.summary(),
            "started": rules.started_s is not None,
            "time_limit_s": rules.time_limit_s,
            "ball_colors": list(rules.ball_colors),
            "tower": _tower(scene),
            "referee_log": rules.log[-40:],
            **self.log.snapshot(),
            "server_time": time.time(),
        }

    # ------------------------------------------------------------------ http
    def _handler(self):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj: Any) -> None:
                self._send(code, json.dumps(obj, default=str).encode(), "application/json")

            def do_GET(self) -> None:  # noqa: N802
                try:
                    path = self.path.split("?")[0]
                    if path in ("/", "/index.html"):
                        return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                    if path == "/api/state":
                        return self._json(200, dash.state())
                    if path == "/api/frame.jpg":
                        return self._send(200, dash.frame_jpeg(), "image/jpeg")
                    self._json(404, {"error": "not found"})
                except Exception as exc:  # noqa: BLE001
                    _log.exception("dashboard GET %s", self.path)
                    self._json(500, {"error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    station = dash.get_station()
                    if self.path == "/api/start":
                        seed = body.get("seed")
                        out = station.start(body.get("own_color") or None, None if seed in (None, "") else int(seed))
                        dash.log.clear()
                        return self._json(200, {"ok": out["ok"], "message": out["message"]})
                    if self.path == "/api/park":
                        out = station.park()
                        return self._json(200, {"ok": out["ok"], "message": out["message"]})
                    if self.path == "/api/agent":
                        import requests

                        r = requests.post(dash.mcp_url, timeout=10, json={
                            "jsonrpc": "2.0", "id": "dashboard", "method": "tools/call",
                            "params": {"name": "agent_send", "arguments": {"message": str(body.get("message", ""))}}})
                        return self._json(200, {"ok": r.ok, "message": r.text[:300]})
                    self._json(404, {"error": "not found"})
                except Exception as exc:  # noqa: BLE001
                    _log.exception("dashboard POST %s", self.path)
                    self._json(200, {"ok": False, "message": str(exc)})

        return Handler


def _tower(scene: dict) -> dict:
    """The tower in inches, from either backend's scene."""
    t = scene.get("tower") or {}
    if "height_in" in t:  # station
        return {"pieces": t.get("placed", []), "height_in": t["height_in"], "pass_in": t.get("pass_height_in", 6.0)}
    if "height_m" in t:  # sim
        return {"pieces": [p.replace(" ", ":") for p in t.get("pieces", [])],
                "height_in": round(t["height_m"] / 0.0254, 2), "pass_in": round(t.get("pass_height_m", 0.152) / 0.0254, 2)}
    return {"pieces": [], "height_in": 0.0, "pass_in": 6.0}
