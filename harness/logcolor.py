"""Colours the run log in a terminal so Claude's verdicts and the stage events stand out (plain text elsewhere)."""

from __future__ import annotations

import logging
import os
import re
import sys

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = (f"\033[{c}m" for c in (31, 32, 33, 34, 35, 36, 37))
ORANGE = "\033[38;5;208m"
GREY = "\033[90m"

# The line's colour, first match wins.
LINE_RULES = (
    ("PASSED", BOLD + GREEN),
    ("Claude does NOT", RED),
    ("Claude sees", BOLD + GREEN),
    ("Claude confirms", BOLD + GREEN),
    ("asking Claude", GREY),
    ("SafetyStop", BOLD + RED),
    ("timed out", YELLOW),
    ("skipping", YELLOW),
    ("switching model", BOLD + CYAN),
    ("arms home", CYAN),
    ("stacked:", BOLD + CYAN),
    ("pool_warming", MAGENTA),
    ("bin:", BOLD + BLUE),
)

# Colour words inside a line (what Claude describes), in their own colours, shown bold.
WORDS = (
    (r"\bgreen\b", GREEN), (r"\b(purple|pink|magenta)\b", MAGENTA), (r"\bblue\b", BLUE),
    (r"\borange\b", ORANGE), (r"\byellow\b", YELLOW), (r"\bwhite\b", WHITE), (r"\bblack\b", GREY),
    (r"\bred\b", RED),
)


def _paint(text: str) -> str:
    base = next((c for key, c in LINE_RULES if key in text), "")
    if not base:
        return text
    out = text
    for pattern, colour in WORDS:
        out = re.sub(pattern, lambda m: f"{BOLD}{colour}{m.group(0)}{RESET}{base}", out, flags=re.I)
    return f"{base}{out}{RESET}"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.levelno >= logging.ERROR:
            return f"{BOLD}{RED}{text}{RESET}"
        return _paint(text)


def install(fmt: str) -> None:
    """Root logging with colours when stderr is a terminal (and NO_COLOR is unset), plain otherwise."""
    handler = logging.StreamHandler()
    use = sys.stderr.isatty() and not os.environ.get("NO_COLOR")
    handler.setFormatter(ColorFormatter(fmt) if use else logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
