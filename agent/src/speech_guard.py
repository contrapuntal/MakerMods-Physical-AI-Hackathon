"""Echo scrubber: the agent's own speech must never contain a freeze word.

The microphone fast path fires an instant robot freeze on "freeze"/"halt", so anything the agent says
(spoken or transcribed) is rewritten to "hold" first. Whole words only, case-insensitive; the first
letter's capitalisation is kept.
"""

import re
from collections.abc import AsyncIterable, AsyncIterator

from livekit.agents.types import TimedString

_MAP = {
    "freeze": "hold",
    "froze": "held",
    "frozen": "held",
    "halt": "hold",
    "halted": "held",
    "halting": "holding",
    "freezes": "holds",
    "freezing": "holding",
    "halts": "holds",
}
# The one trigger-word family (freez*, froze*, halt*), shared with station_monitor's status-text rewriting.
TRIGGER_STEM_RE = re.compile(r"freez\w*|froze\w*|halt\w*", re.IGNORECASE)
_WORD_RE = re.compile(r"(?<![A-Za-z])(%s)(?![A-Za-z])" % "|".join(_MAP), re.IGNORECASE)
_TAIL_RE = re.compile(r"[A-Za-z]+$")
# A trailing partial word is held back only if it could still grow into a trigger word.
_PREFIXES = {w[:i] for w in (*_MAP, "freezer") for i in range(1, len(w) + 1)}


def _sub(m: re.Match) -> str:
    out = _MAP[m.group(1).lower()]
    return out.capitalize() if m.group(1)[0].isupper() else out


def scrub_text(text: str) -> str:
    """Scrub one complete string."""
    return _WORD_RE.sub(_sub, text)


class _Scrubber:
    def __init__(self) -> None:
        self._held = ""

    def feed(self, chunk: str) -> str:
        text = self._held + chunk
        self._held = ""
        m = _TAIL_RE.search(text)
        if m and m.group().lower() in _PREFIXES and not (m.start() > 0 and text[m.start() - 1].isalpha()):
            self._held, text = m.group(), text[: m.start()]
        return scrub_text(text)

    def flush(self) -> str:
        text, self._held = self._held, ""
        return scrub_text(text)


async def scrub_trigger_words(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """Rewrite freeze/halt (and variants) to hold/held across chunk boundaries."""
    s = _Scrubber()
    async for chunk in text:
        out = s.feed(chunk)
        if out:
            yield out
    tail = s.flush()
    if tail:
        yield tail


async def scrub_transcription(text: AsyncIterable[str | TimedString]) -> AsyncIterator[str | TimedString]:
    """Like scrub_trigger_words, but a TimedString (already one word/segment) is scrubbed on its own, keeping its timing."""
    s = _Scrubber()
    async for item in text:
        if isinstance(item, TimedString):
            tail = s.flush()
            if tail:
                yield tail
            new = scrub_text(str(item))
            if new == str(item):
                yield item
            else:
                yield TimedString(
                    new,
                    start_time=item.start_time,
                    end_time=item.end_time,
                    confidence=item.confidence,
                    start_time_offset=item.start_time_offset,
                    speaker_id=item.speaker_id,
                )
        else:
            out = s.feed(item)
            if out:
                yield out
    tail = s.flush()
    if tail:
        yield tail
