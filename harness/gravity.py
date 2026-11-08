"""Scales the Metal arm's gravity compensation (0 = off), per stage if wanted, as ~/dreamcli_gravity.py does for dreamcli.

The demos were recorded with the kit's teleop driver, which sends no gravity torque, so the arm sagged below its
commanded pose: the demos' commanded hand heights sit 4-7 cm above the measured ones near the table, in every
dataset (cables +5.1 cm, stacking +4.1/+4.8, balls +7.2/+5.6). The fine-tunes learned those commanded targets;
Dream's runtime adds gravity torque and reaches them exactly, so every grasp landed several cm high. Scale 0 lets the
arm sag as it did in the demos.

Dream's Damiao driver calls the gravity function on every motor command (`robots.damiao`: each joint's MIT frame
carries the gravity torque at that tick's pose), so `install()` once before the arms are built and `set_scale()` can
then change it at any time: the harness sets each stage's value when the stage's model takes over.

Use it instead of harness/height_offset.py, not with it: both lower the grasp.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)
_original = None
_scale = 1.0


def install(scale: float = 1.0) -> None:
    """Routes every Metal arm built after this call through the scale (looked up when each arm is built)."""
    global _original
    from robots import metal

    if _original is None:
        _original = metal.gravity_nm
        original = _original

        def scaled(q_deg):
            return tuple(_scale * t for t in original(q_deg))

        metal.gravity_nm = scaled
    set_scale(scale)


def set_scale(scale: float) -> None:
    """Gravity torque x `scale` from the next motor command on (1.0 = Dream's own)."""
    global _scale
    if not 0.0 <= scale <= 1.5:
        raise ValueError(f"gravity scale {scale} is outside 0..1.5")
    if scale != _scale:
        _log.info("Metal gravity compensation x%g", scale)
    _scale = scale


def scale() -> float:
    return _scale
