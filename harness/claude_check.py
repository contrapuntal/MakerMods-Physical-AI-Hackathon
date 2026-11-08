"""Claude vision checks for the calls that decide a stage or lose the match.

The colour-and-shape detector (`vision.py`) runs every half second; Claude is asked only at the moments that matter,
with the overhead frame: is a USB cable really out of the strip, is the tower standing on its own, has the black
8-ball landed in the bin. Each answer is structured JSON (`{"yes": bool, "why": str}`).

Calls are blocking HTTP requests, so the executor runs them on a worker thread (`asyncio.to_thread`) and never on the
event loop that drives the arms. Claude Opus 5.5 at low effort keeps them to a few seconds; the server-side refusal
fallback is on, so a rare safety decline is retried on another model inside the same call.

The key is read from ~/team-harness/.anthropic_key (written by save_anthropic_key.sh), or ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib

import anthropic
import cv2
import numpy as np
import numpy.typing as npt

_log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
KEY_FILE = pathlib.Path("~/team-harness/.anthropic_key").expanduser()

SYSTEM = (
    "You check one robot-arm table from its overhead camera for a timed robotics match. Answer only from what the "
    "image shows. When the image is ambiguous (an arm or a hand hides the thing asked about), answer no and say so."
)

QUESTIONS = {
    "unplugged_since": (
        "Image 1 is the table at the start of the run; image 2 is the same table now. Compare the orange USB "
        "cables at the black power strip. Has at least one cable that was plugged into the strip in image 1 been "
        "pulled out of its port in image 2? A cable that was already unplugged in image 1 does not count."
    ),
    "unplugged": (
        "On the black power strip, orange USB cables are plugged into its USB ports. Is at least one orange USB "
        "cable's plug completely out of its port now, with its metal connector visibly away from the strip?"
    ),
    "tower": (
        "Is there a free-standing stack of at least three foam cubes (stacked on top of each other, not side by "
        "side), standing on its own with no gripper touching it?"
    ),
    "eight_ball_in_bin": "Is a black ball inside the white basket?",
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"yes": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["yes", "why"],
    "additionalProperties": False,
}


def _load_key() -> str | None:
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
        if key:
            return key
    return os.environ.get("ANTHROPIC_API_KEY") or None


class ClaudeVerifier:
    """Asks Claude yes/no questions about an overhead frame. `available` is False without a key."""

    def __init__(self) -> None:
        key = _load_key()
        self.available = key is not None
        self._client = anthropic.Anthropic(api_key=key, max_retries=1, timeout=20.0) if key else None

    def probe(self) -> str | None:
        """One tiny request before the run: None when Claude answers, else why it can't (and the checks go off).

        A key with no credit, or a rejected one, fails here in under a second instead of at a stage's pass bar."""
        if self._client is None:
            return "no Anthropic key"
        try:
            self._client.messages.create(model=MODEL, max_tokens=16, output_config={"effort": "low"},
                                         messages=[{"role": "user", "content": "Reply with: ok"}])
            return None
        except anthropic.APIStatusError as exc:
            why = f"{exc.status_code}: {getattr(exc, 'message', exc)}"
        except anthropic.APIConnectionError:
            why = "network error"
        self._client, self.available = None, False
        return why

    def ask(self, question: str, rgb: npt.NDArray[np.uint8],
            before: npt.NDArray[np.uint8] | None = None) -> tuple[bool | None, str]:
        """One yes/no answer about the frame (and, for a before/after question, the frame at go first); None means
        no opinion (Claude unreachable or out of credit), so the caller falls back to the colour checks."""
        if self._client is None:
            return None, "no Anthropic key"
        images = []
        for frame in ([before] if before is not None else []) + [rgb]:
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None, "could not encode the frame"
            images.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.standard_b64encode(jpg.tobytes()).decode()}})
        try:
            response = self._client.beta.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": [
                    *images, {"type": "text", "text": QUESTIONS.get(question, question)},
                ]}],
            )
        except anthropic.RateLimitError:
            return None, "rate limited"
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:  # a rejected key, no credit, a bad request: it won't fix itself mid-match
                _log.error("claude: %s %s; vision checks are off", exc.status_code, getattr(exc, "message", exc))
                self._client, self.available = None, False
            return None, f"API error {exc.status_code}"
        except anthropic.APIConnectionError:
            return None, "network error"
        if response.stop_reason == "refusal":
            return None, "refused"
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
            return bool(data["yes"]), str(data.get("why", ""))[:200]
        except (ValueError, KeyError, TypeError):
            return None, f"unreadable answer: {text[:80]!r}"
