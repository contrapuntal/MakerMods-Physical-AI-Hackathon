"""Async client for the Dream station HTTP API (see robots/station_api.py).

Dry-run by default: start/home only make a request when STATION_LIVE=1.
status() and stop() always make requests (stop must never be blocked).
/reset is deliberately not implemented: it lets the arms go limp.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger("station_client")

DEFAULT_URL = "http://127.0.0.1:8091"
# The station waits up to 60 s for the controller (station_api.REQUEST_TIMEOUT_S);
# stop/home can legitimately take a few seconds. Status should be fast.
STATUS_TIMEOUT_S = 5.0
MOTION_TIMEOUT_S = 70.0
# Stop must not hang on a half-open SSH tunnel. The station cancels the run the moment /stop arrives,
# so a short client timeout loses nothing. Module globals so tests can patch them.
STOP_TOTAL_S = 8.0
STOP_CONNECT_S = 2.0


class StationError(Exception):
    """The station answered with an error, or could not be reached.

    Attributes: status (HTTP status, None if unreachable), message (station's error text).
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status

    @property
    def refused(self) -> bool:
        """409: the station refused the transition (e.g. stop during a home move)."""
        return self.status == 409


@dataclass
class StationResult:
    """Outcome of a call. `dry_run` True means nothing was sent; `sent` is what would have been."""

    dry_run: bool
    method: str
    path: str
    sent: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)


def is_live() -> bool:
    return os.environ.get("STATION_LIVE") == "1"


class StationClient:
    def __init__(self, base_url: str | None = None, live: bool | None = None) -> None:
        self.base_url = (base_url or os.environ.get("STATION_URL") or DEFAULT_URL).rstrip("/")
        self._live = live  # None: read STATION_LIVE at call time

    @property
    def live(self) -> bool:
        return is_live() if self._live is None else self._live

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        timeout_s: float,
        connect_s: float | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=connect_s)
            async with (
                aiohttp.ClientSession(timeout=timeout) as s,
                s.request(method, url, json=body) as resp,
            ):
                try:
                    data = await resp.json(content_type=None)
                except ValueError:  # JSONDecodeError / bad payload
                    data = {"error": f"non-JSON reply (HTTP {resp.status})"}
                if not isinstance(data, dict):
                    data = {"error": f"unexpected reply (HTTP {resp.status})"}
                if resp.status >= 400:
                    raise StationError(str(data.get("error", f"HTTP {resp.status}")), resp.status)
                return data
        except StationError:
            raise
        except TimeoutError as exc:
            raise StationError(f"station did not answer within {timeout_s:g} s") from exc
        except aiohttp.ClientError as exc:
            raise StationError(f"station unreachable: {exc}") from exc

    async def _motion(self, path: str, body: dict[str, Any] | None) -> StationResult:
        if not self.live:
            logger.info("DRY-RUN POST %s %s (set STATION_LIVE=1 to send)", path, body)
            return StationResult(True, "POST", path, body)
        data = await self._request("POST", path, body if body is not None else {}, MOTION_TIMEOUT_S)
        return StationResult(False, "POST", path, body, data)

    async def status(self) -> dict[str, Any]:
        """GET /status. Read-only, always sent."""
        return await self._request("GET", "/status", None, STATUS_TIMEOUT_S)

    async def start(self, prompt: str, checkpoint: str) -> StationResult:
        """POST /start {"prompt","checkpoint"}. Moves the arms."""
        return await self._motion("/start", {"prompt": prompt, "checkpoint": checkpoint})

    async def home(self) -> StationResult:
        """POST /home. Moves the arms home."""
        return await self._motion("/home", None)

    async def stop(self) -> StationResult:
        """POST /stop. Always sent, even in dry-run. Raises StationError (refused=True on 409)."""
        data = await self._request("POST", "/stop", {}, STOP_TOTAL_S, STOP_CONNECT_S)
        return StationResult(False, "POST", "/stop", None, data)
