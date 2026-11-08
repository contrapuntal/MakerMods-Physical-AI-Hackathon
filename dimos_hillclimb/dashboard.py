"""The match dashboard: one web page beside the HillClimbStation module, for the sim and the real station alike.

    http://127.0.0.1:8770/          the page (dashboard.html)
    GET  /api/state                 referee summary, tower, bin, the skill-call log, backend
    GET  /api/frame.jpg             the overhead frame (on the station: with the server's detections drawn on)
    GET  /api/cameras               camera names, YOLO availability, the station's --augment keys
    GET  /api/cam/<name>.jpg?modes=box,suppress&target=green&detector=both
                                    one camera, augmented here as harness/augment.py would for Dream (modes=off: raw)
    GET  /api/sent/<key>.jpg        on the station with --augment: the exact frame last sent to Dream
    POST /api/start  {own_color, seed}   a new match (as the start_match skill)
    POST /api/park                  both arms to rest / park
    POST /api/agent  {message}      text to DimOS's agent (through the MCP server's agent_send), e.g. the go

It lives in the skills module because that is where the referee (stage, score, clock) and the backend both are; the
station server alone doesn't know the score. Standard library only.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np

_log = logging.getLogger(__name__)

PAGE = pathlib.Path(__file__).with_name("dashboard.html")
YOLO_WEIGHTS = os.environ.get("HILLCLIMB_YOLO", "yolo11n.pt")  # a path, or a name ultralytics downloads
# The real cameras from the hackathon kit's station dashboard (dhk-collect or scripts/check_cameras.py), reached
# through `ssh -L 8080:localhost:8080 pah@<station>`: set to http://localhost:8080 to show and augment those instead
# of the backend's cameras.
CAMERAS_URL = os.environ.get("HILLCLIMB_CAMERAS_URL", "").rstrip("/")


class KitCameras:
    """The kit dashboard's cameras: GET /api/status lists the roles, GET /api/camera/<role>.jpg gives a frame."""

    def __init__(self, url: str) -> None:
        import requests

        self.url = url
        self._local = threading.local()  # one connection per request thread: a Session is not thread-safe
        self._names: tuple[float, list[str]] | None = None
        self._last: dict[str, np.ndarray] = {}  # role -> the last frame that came through
        self.error: str | None = None

    @property
    def http(self):
        import requests

        if not hasattr(self._local, "http"):
            self._local.http = requests.Session()
        return self._local.http

    def camera_names(self) -> list[str]:
        if self._names and time.monotonic() - self._names[0] < 5.0:
            return self._names[1]
        try:
            # 10 s: the status took 3.3 s over the tunnel while dhk-collect recorded
            names = list(self.http.get(f"{self.url}/api/status", timeout=10).json().get("cameras", []))
            self.error = None
        except Exception as exc:  # noqa: BLE001 - the tunnel is down or the kit dashboard isn't running
            names, self.error = [], f"no answer from {self.url} ({type(exc).__name__}): is dhk-collect or " \
                                    "check_cameras.py running on the station, with the ssh -L 8080 tunnel up?"
        self._names = (time.monotonic(), names)
        return names

    def camera_rgb(self, role: str) -> np.ndarray | None:
        import cv2

        # Over the SSH tunnel, while dhk-collect records, a frame takes 0.6-2.7 s: wait for it, and on a miss show the
        # last frame that came through rather than a black one.
        try:
            r = self.http.get(f"{self.url}/api/camera/{role}.jpg", timeout=8)
            bgr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) if r.ok else None
        except Exception:  # noqa: BLE001
            bgr = None
        if bgr is not None:
            self._last[role] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._last.get(role)
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
        self._cams: dict[tuple, tuple[float, bytes, dict]] = {}  # (camera, modes, target, detector) -> frame
        self._cam_lock = threading.Lock()
        self._finders: dict[str, Any] = {}
        self._yolo_error: str | None = None
        self.kit = KitCameras(CAMERAS_URL) if CAMERAS_URL else None
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

    # ------------------------------------------------------------------ cameras, with the augmentation
    def _camera_source(self):
        return self.kit if self.kit is not None else self.get_station()

    def cameras(self) -> dict:
        station = self.get_station()
        if self.kit is not None:
            info = {"cameras": self.kit.camera_names(), "source": f"station cameras via {self.kit.url}",
                    "source_error": self.kit.error}
        else:
            try:
                info = station.camera_info() if hasattr(station, "camera_info") else {"cameras": station.camera_names()}
            except Exception as exc:  # noqa: BLE001 - the station server is down
                info = {"cameras": [], "source_error": f"station server: {exc}"}
            info["source"] = "sim cameras" if self.backend == "sim" else "station server cameras"
        return {**info, "yolo": self._yolo_error is None and _has_ultralytics(), "yolo_error": self._yolo_error,
                "own_color": station.rules.own_color}

    def _finder(self, detector: str):
        from harness import augment
        from harness.config import DEFAULT_CALIBRATION, Calibration

        if detector not in self._finders:
            cal = Calibration.load(DEFAULT_CALIBRATION)
            try:
                self._finders[detector] = augment.Finder(cal, detector, YOLO_WEIGHTS)
            except Exception as exc:  # noqa: BLE001 - no ultralytics or no weights: say so, fall back to colour
                self._yolo_error = str(exc)
                self._finders[detector] = augment.Finder(cal, "hsv")
        return self._finders[detector]

    def cam_jpeg(self, name: str, modes: str, target: str, detector: str, yolo_boxes: bool = False
                 ) -> tuple[bytes, dict]:
        import cv2

        from harness import augment

        key = (name, modes, target, detector, yolo_boxes)
        with self._cam_lock:
            hit = self._cams.get(key)
            if hit and time.monotonic() - hit[0] < FRAME_MAX_AGE_S:
                return hit[1], hit[2]
        rgb = self._camera_source().camera_rgb(name)
        if rgb is None:
            rgb = np.zeros((360, 640, 3), np.uint8)
        info: dict = {"found": 0, "ms": 0, "sources": "", "yolo_balls": "", "yolo_colors": ""}
        mode_list = tuple(m for m in modes.split(",") if m in augment.MODES)
        if mode_list or yolo_boxes:
            finder = self._finder(detector)
            cal_color = next((c for c in finder.cal.ball_colors if augment.spoken(c) == target), target)
            t0 = time.perf_counter()
            raw = finder.yolo.detect_all(rgb) if finder.yolo is not None else []
            balls = finder(rgb, cal_color, raw if finder.yolo is not None else None) \
                if cal_color in finder.cal.colors else []
            if mode_list:
                rgb = augment.annotate(rgb, balls, mode_list, cal_color, finder.cal)
            named = augment.label_balls(rgb, raw, finder.cal)  # each YOLO ball's colour, read from its pixels
            if yolo_boxes:
                rgb = augment.draw_yolo_boxes(rgb, named)
            info = {"found": len(balls), "ms": round((time.perf_counter() - t0) * 1000),
                    "sources": ",".join(sorted({b.source for b in balls})),
                    "yolo_balls": sum(c not in augment.DECOY_CLASSES for c, _, _ in named),  # balls, not decoys
                    "yolo_colors": ",".join(augment.shown_name(c) for c, _, _ in named)}
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._cam_lock:
            self._cams[key] = (time.monotonic(), jpg.tobytes(), info)
        return jpg.tobytes(), info

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

            def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the page gave up on a slow frame and asked for a newer one

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
                    if path == "/api/cameras":
                        return self._json(200, dash.cameras())
                    if path.startswith("/api/cam/") and path.endswith(".jpg"):
                        q = parse_qs(urlsplit(self.path).query)
                        arg = lambda k, d: (q.get(k) or [d])[0]  # noqa: E731
                        jpg, info = dash.cam_jpeg(path[len("/api/cam/"):-4], arg("modes", "off"),
                                                  arg("target", "green"), arg("detector", "both"),
                                                  arg("boxes", "0") == "1")
                        return self._send(200, jpg, "image/jpeg",
                                          {f"X-{k.replace('_', '-').title()}": str(v) for k, v in info.items()})
                    if path.startswith("/api/sent/") and path.endswith(".jpg"):
                        station = dash.get_station()
                        jpg = station.sent_jpeg(path[len("/api/sent/"):-4]) if hasattr(station, "sent_jpeg") else None
                        if jpg is None:
                            return self._json(404, {"error": "nothing sent under that key"})
                        return self._send(200, jpg, "image/jpeg")
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


def _has_ultralytics() -> bool:
    import importlib.util
    return importlib.util.find_spec("ultralytics") is not None


def _tower(scene: dict) -> dict:
    """The tower in inches, from either backend's scene."""
    t = scene.get("tower") or {}
    if "height_in" in t:  # station
        return {"pieces": t.get("placed", []), "height_in": t["height_in"], "pass_in": t.get("pass_height_in", 6.0)}
    if "height_m" in t:  # sim
        return {"pieces": [p.replace(" ", ":") for p in t.get("pieces", [])],
                "height_in": round(t["height_m"] / 0.0254, 2), "pass_in": round(t.get("pass_height_m", 0.152) / 0.0254, 2)}
    return {"pieces": [], "height_in": 0.0, "pass_in": 6.0}
