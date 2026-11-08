"""A height correction for a fine-tune that grasps too high: lowers the model's own hand targets near the table.

The cable model reached the plug's x/y but never went below ~12.8 cm, where its 52 demos closed at a median 7.3 cm.
A Metal fine-tune answers tool poses (`action.<side>_eef_xyz`, metres, in the arm's base frame), and Dream's client
solves them to joints in `dream.client_ik.ClientIK.solve()` with dream's own IK, speed caps and joint limits. This
wraps that method: before the solve, each target's z is lowered by up to `drop_m`, so the joints still come from
Dream's solver and its limits.

The drop fades in with height: none above `engage_m`, the full `drop_m` at and below `full_m`, linear between, so
moves high above the table are unchanged and the hand path stays smooth. The span is kept wider than the drop, so a
higher target never maps below a lower one. A target is never pushed below `floor_m` (where the demos' lowest 1%
sat); one the model itself sends lower is left as it is.

Only the models and sides named in `install()` are touched; every other client runs as Dream wrote it.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Drop:
    drop_m: float  # how far to lower a target at or below full_m
    engage_m: float = 0.28  # no change above this height
    full_m: float = 0.17  # the full drop at and below this height (the cable model closed at 10-17 cm uncorrected)
    floor_m: float = 0.04  # never push a target below this

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_m <= 0.10:
            raise ValueError(f"drop {self.drop_m} m is outside 0..0.10 m")
        if self.engage_m - self.full_m <= self.drop_m:
            raise ValueError("the fade span must be wider than the drop, or a higher target could land lower")

    def apply(self, z: np.ndarray) -> np.ndarray:
        """The lowered heights for target heights `z` (metres)."""
        z = np.asarray(z, dtype=np.float64)
        ramp = np.clip((self.engage_m - z) / (self.engage_m - self.full_m), 0.0, 1.0)
        lowered = z - self.drop_m * ramp
        return np.maximum(lowered, np.minimum(z, self.floor_m))


_lock = threading.Lock()
_rules: dict[tuple[str, str], Drop] = {}  # (model id, side) -> drop
_original = None
_stats: dict[str, float] = {}


def install(model: str, drop: Drop, sides: tuple[str, ...] = ("right",)) -> None:
    """Lowers `model`'s hand targets on `sides` by `drop` from now on, in this process."""
    global _original
    from dream import client_ik

    with _lock:
        for side in sides:
            _rules[(model, side)] = drop
        if _original is None:
            _original = client_ik.ClientIK.solve

            def solve(self, action, *args, **kwargs):  # noqa: ANN001 - dream's own signature
                return _original(self, _lowered(getattr(self, "_model_id", None), action), *args, **kwargs)

            client_ik.ClientIK.solve = solve
    _log.info("height correction on %s %s: down %.1f cm below %.0f cm (full below %.0f cm), floor %.1f cm",
              model, "/".join(sides), drop.drop_m * 100, drop.engage_m * 100, drop.full_m * 100, drop.floor_m * 100)


def uninstall() -> None:
    global _original
    from dream import client_ik

    with _lock:
        if _original is not None:
            client_ik.ClientIK.solve = _original
            _original = None
        _rules.clear()


def stats() -> dict[str, float]:
    """The lowest target seen before and after the correction, per side, in cm (for the run report)."""
    return dict(_stats)


def _lowered(model: str | None, action: dict[str, Any]) -> dict[str, Any]:
    per_key = action.get("per_key") if isinstance(action, dict) else None
    if model is None or not isinstance(per_key, dict):
        return action
    changed = None
    for (m, side), drop in _rules.items():
        key = f"action.{side}_eef_xyz"
        if m != model or key not in per_key:
            continue
        try:
            rows = np.array(per_key[key], dtype=np.float64)
        except (TypeError, ValueError):
            continue  # let Dream's own checks refuse it
        if rows.ndim != 2 or rows.shape[1] != 3:
            continue
        before = float(rows[:, 2].min())
        rows[:, 2] = drop.apply(rows[:, 2])
        if changed is None:
            changed = dict(per_key)
        changed[key] = rows
        _stats[f"{side}_min_target_cm"] = min(_stats.get(f"{side}_min_target_cm", 1e9), before * 100)
        _stats[f"{side}_min_lowered_cm"] = min(_stats.get(f"{side}_min_lowered_cm", 1e9), float(rows[:, 2].min()) * 100)
    if changed is None:
        return action
    out = dict(action)
    out["per_key"] = changed
    return out
