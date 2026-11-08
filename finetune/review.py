"""The label review page: go through the auto-labelled frames, fix the boxes, mark each frame reviewed.

    repos/dimos/.venv/bin/python -m finetune review        # http://127.0.0.1:8790/

GET  /                    the page (review.html)
GET  /api/frames          every frame: name, split, status, camera, episode, box count
GET  /api/frame/<name>    the JPEG
GET  /api/labels/<name>   {"status", "split", "boxes"}
POST /api/labels/<name>   {"status": ok|skip|auto, "boxes": [...]}   saves at once (labels.json, atomically)
GET  /api/stats           counts per split and status, reviewed boxes per class
Standard library only; one lock around labels.json.
"""

from __future__ import annotations

import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote

from finetune import data

PAGE = pathlib.Path(__file__).with_name("review.html")


def serve(host: str = "127.0.0.1", port: int = 8790, root: pathlib.Path = data.ROOT) -> None:
    lock = threading.Lock()
    labels = data.load_labels(root)
    if not labels:
        raise SystemExit("no frames yet: run `extract` and `label` first")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _name(self, prefix: str) -> str | None:
            name = unquote(self.path.split("?")[0][len(prefix):])
            return name if name in labels and "/" not in name else None

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/frames":
                with lock:
                    rows = [{"name": n, "split": r.get("split"), "status": r.get("status"), "n": len(r["boxes"]),
                             **data.frame_info(n)} for n, r in sorted(labels.items())]
                return self._json(200, {"frames": rows, "classes": data.CLASSES})
            if path == "/api/stats":
                with lock:
                    data.save_labels(labels, root)
                return self._json(200, data.stats(root))
            if path.startswith("/api/frame/"):
                name = self._name("/api/frame/")
                if name is None:
                    return self._json(404, {"error": "no such frame"})
                return self._send(200, (root / "frames" / name).read_bytes(), "image/jpeg")
            if path.startswith("/api/labels/"):
                name = self._name("/api/labels/")
                if name is None:
                    return self._json(404, {"error": "no such frame"})
                with lock:
                    return self._json(200, labels[name])
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/api/labels/"):
                return self._json(404, {"error": "not found"})
            name = self._name("/api/labels/")
            if name is None:
                return self._json(404, {"error": "no such frame"})
            if self.headers.get("X-Review") != "1":  # only the page itself writes (no other site through the browser)
                return self._json(403, {"error": "missing X-Review header"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            status = body.get("status")
            boxes = body.get("boxes")
            if status not in ("ok", "skip", "auto") or not isinstance(boxes, list):
                return self._json(400, {"error": "need status ok|skip|auto and a boxes list"})
            clean = []
            for b in boxes:
                if b.get("cls") not in data.CLASSES or len(b.get("box", [])) != 4:
                    return self._json(400, {"error": f"bad box {b}"})
                x0, y0, x1, y1 = (float(v) for v in b["box"])
                clean.append({"cls": b["cls"], "box": [round(min(x0, x1), 1), round(min(y0, y1), 1),
                                                       round(max(x0, x1), 1), round(max(y0, y1), 1)],
                              "src": b.get("src", "human"), "score": b.get("score", 1.0)})
            with lock:
                labels[name].update(status=status, boxes=clean)
                data.save_labels(labels, root)
            self._json(200, {"ok": True})

    http = ThreadingHTTPServer((host, port), Handler)
    print(f"label review on http://{host}:{port}/  ({len(labels)} frames). Ctrl-C stops; every save is on disk.")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
