"""The world model: every object on the table, found in the overhead frame by colour and shape.

Fast and deterministic on purpose: it runs before every step of a 5-minute match. Each colour is an HSV box from the
calibration; a blob's shape tells a cube (squat) from a rod (long and thin) from a ball (small and round). Everything
is in the overhead frame's pixels; distances use the calibration's px_per_cm on the table plane.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import numpy.typing as npt

from harness.config import Calibration, ColorRange

CUBE_MAX_ELONG = 1.75  # a cube is about square from above (1.1), ~1.55 lying on its side; a standing rod is 2.1
ROD_LYING_ELONG = 2.8  # a rod lying flat is long and thin; standing up it looks squat
EIGHT_AREA = (100, 500)  # px^2, the black 8-ball as a whole ball
BALL_MIN_ROUND = 0.45  # 4*pi*area/perimeter^2: 1 for a disc; a half-hidden ball measures ~0.5
BALL_MAX_ELONG = 2.4  # a ball half hidden by the bin's rim looks lopsided (2.0 measured)


@dataclasses.dataclass
class Obj:
    """One object found in the frame."""

    kind: str  # cube | rod | ball | eight_ball | plug
    color: str
    x: float  # centroid, px
    y: float
    area: float  # px^2
    elongation: float  # long side / short side of its rotated box
    box: tuple[int, int, int, int]  # x, y, w, h
    brightness: float = 0.0  # mean HSV value (0-255) over the blob: the stacking order's darkest-to-lightest key

    def dist_cm(self, other: "Obj", px_per_cm: float) -> float:
        return math.hypot(self.x - other.x, self.y - other.y) / px_per_cm

    @property
    def lying(self) -> bool:
        """A rod lying flat on the table (as opposed to standing upright)."""
        return self.kind == "rod" and self.elongation >= ROD_LYING_ELONG

    def label(self) -> str:
        if self.kind == "rod":
            return f"{self.color} rod ({'lying' if self.lying else 'upright'})"
        return f"{self.color} {self.kind}"


@dataclasses.dataclass
class World:
    """Every object found in one overhead frame, plus where the fixed regions are."""

    objects: list[Obj]
    cal: Calibration
    timestamp_s: float = 0.0
    cable_px_in_hub: int = 0  # cable-coloured pixels at the hub's ports: drops when a plug is pulled out

    def of(self, kind: str, color: str | None = None) -> list[Obj]:
        return [o for o in self.objects if o.kind == kind and (color is None or o.color == color)]

    def inside(self, roi: list[tuple[int, int]], objs: list[Obj]) -> list[Obj]:
        if len(roi) < 3:
            return []
        poly = np.array(roi, dtype=np.int32)
        return [o for o in objs if cv2.pointPolygonTest(poly, (o.x, o.y), False) >= 0]

    def outside(self, roi: list[tuple[int, int]], objs: list[Obj]) -> list[Obj]:
        ins = {id(o) for o in self.inside(roi, objs)}
        return [o for o in objs if id(o) not in ins]

    # -- stage 1
    def plugs_in_hub(self) -> list[Obj]:
        return self.inside(self.cal.hub_roi, self.of("plug"))

    def plugs_off_hub(self) -> list[Obj]:
        """Cable-coloured blobs just off the strip: a plug pulled out lands here."""
        return self.outside(self.cal.hub_roi, self.of("plug"))

    # -- stage 3
    def balls_in_bin(self, color: str) -> list[Obj]:
        return self.inside(self.cal.bin_roi, self.of("ball", color))

    def balls_on_table(self, color: str) -> list[Obj]:
        return self.outside(self.cal.bin_roi, self.of("ball", color))

    def eight_ball_in_bin(self) -> bool:
        return bool(self.inside(self.cal.bin_roi, self.of("eight_ball")))

    def avoid_in_bin(self) -> list[Obj]:
        return self.inside(self.cal.bin_roi, self.of("avoid_ball"))

    # -- stage 2
    def loose(self, kind: str) -> list[Obj]:
        return self.outside(self.cal.tower_roi, self.of(kind))

    def at_tower(self, kind: str) -> list[Obj]:
        return self.inside(self.cal.tower_roi, self.of(kind))

    def nearest_cm(self, target: Obj, others: list[Obj]) -> float:
        d = [target.dist_cm(o, self.cal.px_per_cm) for o in others if o is not target]
        return min(d) if d else float("inf")

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.label()] = counts.get(o.label(), 0) + 1
        text = ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "nothing"
        return f"{text}; {self.cable_px_in_hub} cable px at the hub"


def mask_for(hsv: npt.NDArray[np.uint8], c: ColorRange) -> npt.NDArray[np.uint8]:
    lo_s, hi_s = (c.s_lo, c.v_lo), (c.s_hi, c.v_hi)
    if c.h_lo <= c.h_hi:
        return cv2.inRange(hsv, (c.h_lo, *lo_s), (c.h_hi, *hi_s))
    return cv2.inRange(hsv, (c.h_lo, *lo_s), (179, *hi_s)) | cv2.inRange(hsv, (0, *lo_s), (c.h_hi, *hi_s))


def _blobs(mask: npt.NDArray[np.uint8], min_area: float, value: npt.NDArray[np.uint8] | None = None) -> list[tuple]:
    """(cx, cy, area, elongation, box, brightness, roundness) for each blob of at least min_area."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        elong = max(w, h) / max(1.0, min(w, h))
        bright = 0.0
        if value is not None:
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            bright = float(cv2.mean(value, mask=blob)[0])
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area, elong, cv2.boundingRect(c), bright,
                    4 * math.pi * area / max(1.0, cv2.arcLength(c, True)) ** 2))
    return out


def detect(rgb: npt.NDArray[np.uint8], cal: Calibration, timestamp_s: float = 0.0) -> World:
    """Finds every calibrated object in an RGB overhead frame (resized to the calibration's frame size)."""
    w, h = cal.frame_size
    if rgb.shape[1] != w or rgb.shape[0] != h:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    work = np.zeros((h, w), np.uint8)
    if len(cal.work_roi) >= 3:
        cv2.fillPoly(work, [np.array(cal.work_roi, np.int32)], 255)
    else:
        work[:] = 255
    for poly in cal.ignore_rois:
        cv2.fillPoly(work, [np.array(poly, np.int32)], 0)
    objs: list[Obj] = []

    def add(kind: str, color: str, blob) -> None:
        cx, cy, area, elong, box, bright = blob[:6]
        objs.append(Obj(kind, color, cx, cy, area, elong, box, bright))

    for color in set(cal.cube_colors) | set(cal.rod_colors):
        mask = mask_for(hsv, cal.colors[color]) & work
        for b in _blobs(mask, min(cal.cube_area[0], cal.rod_area[0]), hsv[:, :, 2]):
            area, elong = b[2], b[3]
            # Measured on the layout: cubes 1.08-1.16 long-to-short, a standing rod 2.1, a lying rod 3.45.
            if color in cal.rod_colors and elong >= CUBE_MAX_ELONG and cal.rod_area[0] <= area <= cal.rod_area[1]:
                add("rod", color, b)
            elif color in cal.cube_colors and elong < CUBE_MAX_ELONG and cal.cube_area[0] <= area <= cal.cube_area[1]:
                add("cube", color, b)
    balls_area = work.copy()  # no ball sits in the hub (its white box would read as a white ball)
    if len(cal.hub_roi) >= 3:
        cv2.fillPoly(balls_area, [np.array(cal.hub_roi, np.int32)], 0)
    for kind, group in (("ball", cal.ball_colors), ("avoid_ball", cal.avoid_balls)):
        for color in group:
            mask = mask_for(hsv, cal.colors[color]) & balls_area
            for b in _blobs(mask, cal.ball_area[0]):
                if b[2] <= cal.ball_area[1] and b[3] < BALL_MAX_ELONG and b[6] >= BALL_MIN_ROUND:
                    add(kind, color, b)
    eight = mask_for(hsv, cal.colors[cal.eight_ball]) & balls_area
    for b in _blobs(eight, cal.ball_area[0]):
        # The 8-ball must look like a whole ball: dark corners of the strip or a gripper are not round.
        # A 40 mm ball is ~17 px across here (~230 px^2): a phone or a gripper is not.
        if EIGHT_AREA[0] <= b[2] <= EIGHT_AREA[1] and b[3] < 1.5 and b[6] >= 0.7:
            add("eight_ball", cal.eight_ball, b)
    cable_px = 0
    if len(cal.hub_roi) >= 3:
        ports = np.zeros((h, w), np.uint8)
        cv2.fillPoly(ports, [np.array(cal.hub_roi, np.int32)], 255)
        for color in cal.cable_colors:
            cable_px += int(cv2.countNonZero(mask_for(hsv, cal.colors[color]) & ports))
    hub = np.zeros((h, w), np.uint8)
    if len(cal.hub_roi) >= 3:
        # Plugs are looked for around the hub (generously), so one pulled a few cm out still counts as found.
        x, y, bw, bh = cv2.boundingRect(np.array(cal.hub_roi, np.int32))
        pad = int(6 * cal.px_per_cm)
        hub[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad] = 255
    for color in cal.cable_colors:
        mask = mask_for(hsv, cal.colors[color]) & hub
        for b in _blobs(mask, cal.plug_area[0]):
            if b[2] <= cal.plug_area[1]:
                add("plug", color, b)
    return World(objs, cal, timestamp_s, cable_px)


@dataclasses.dataclass
class Layout:
    """Where the bin and the strip's USB ports are in this frame, found from the image itself."""

    bin_roi: list[tuple[int, int]] | None
    hub_roi: list[tuple[int, int]] | None

    def note(self) -> str:
        return (f"bin {'found' if self.bin_roi else 'NOT found (calibrated spot)'}, "
                f"strip {'found' if self.hub_roi else 'NOT found (calibrated spot)'}")


def locate_layout(rgb: npt.NDArray[np.uint8], cal: Calibration) -> Layout:
    """Finds the bin and the strip wherever the judges put them, so nothing needs taping.

    - The bin (the white basket) is the largest pale region with a dense mesh texture: the table is pale too, but
      smooth. Keyboards at the table's edges are darker and touch the frame's sides, so they are left out.
    - The strip is a long dark bar on the table; of the dark bars, the one with the most orange cable pixels on and
      around it is the strip (its USB ports are where the cables plug in). Its box becomes the hub region.
    Either can fail (an odd angle, clutter): the calibrated spot is then kept.
    """
    w, h = cal.frame_size
    if rgb.shape[1] != w or rgb.shape[0] != h:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    table = np.zeros((h, w), np.uint8)
    top = min(p[1] for p in cal.work_roi) if cal.work_roi else 0
    table[top:, 8:w - 8] = 255
    for poly in cal.ignore_rois:
        cv2.fillPoly(table, [np.array(poly, np.int32)], 0)

    # The bin.
    pale = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 130)  # 130, not 150: the basket's shadowed side is darker
    dens = cv2.boxFilter((cv2.Canny(gray, 30, 100) > 0).astype(np.float32), -1, (15, 15))
    mesh = ((dens > 0.10) & pale).astype(np.uint8) * 255 & table
    mesh = cv2.morphologyEx(mesh, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    mesh = cv2.morphologyEx(mesh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mesh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bins = sorted([c for c in contours if 2500 < cv2.contourArea(c) < 30000], key=cv2.contourArea, reverse=True)
    bin_roi = None
    if bins:
        # The hull fills the basket's smooth floor, which the mesh test leaves out. If the hull is far bigger than
        # the outline, clutter has merged in: keep the outline itself then.
        outline = bins[0]
        hull = cv2.convexHull(outline)
        shape = hull if cv2.contourArea(hull) < 1.8 * cv2.contourArea(outline) else outline
        bin_roi = [tuple(int(v) for v in p[0]) for p in cv2.approxPolyDP(shape, 4, True)]

    # The strip.
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 55)) & table
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cable = np.zeros((h, w), np.uint8)
    for color in cal.cable_colors:
        cable |= mask_for(hsv, cal.colors[color])
    best, best_px = None, 0
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    work_poly = np.array(cal.work_roi, np.int32) if len(cal.work_roi) >= 3 else None
    for c in contours:
        area = cv2.contourArea(c)
        if not 700 <= area <= 6000:  # a laptop is far bigger than the strip (~2,000 px here)
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if not 1.7 < max(rw, rh) / max(1.0, min(rw, rh)) < 6 or min(rw, rh) > 55:
            continue
        if work_poly is not None and cv2.pointPolygonTest(work_poly, (cx, cy), False) < 0:
            continue  # off the working table (keyboards and laptops at its edges)
        x, y, bw, bh = cv2.boundingRect(c)
        pad = 6
        px = int(cv2.countNonZero(cable[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]))
        if px > best_px:
            best, best_px = (x - pad, y - pad, bw + 2 * pad, bh + 2 * pad), px
    hub_roi = None
    if best is not None and best_px >= 20:
        x, y, bw, bh = best
        hub_roi = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
    return Layout(bin_roi, hub_roi)


def with_layout(cal: Calibration, layout: Layout) -> Calibration:
    """The calibration with the regions this frame found (the calibrated ones kept where nothing was found)."""
    return dataclasses.replace(cal, bin_roi=layout.bin_roi or cal.bin_roi, hub_roi=layout.hub_roi or cal.hub_roi)


def draw(rgb: npt.NDArray[np.uint8], world: World) -> npt.NDArray[np.uint8]:
    """The frame (BGR) with the regions and every detection drawn on, for tuning the calibration."""
    w, h = world.cal.frame_size
    img = cv2.cvtColor(cv2.resize(rgb, (w, h)), cv2.COLOR_RGB2BGR)
    for roi, col, name in ((world.cal.work_roi, (180, 180, 180), "work"), (world.cal.hub_roi, (0, 200, 255), "hub"),
                           (world.cal.bin_roi, (255, 120, 0), "bin"), (world.cal.tower_roi, (0, 255, 0), "tower")):
        if len(roi) >= 3:
            cv2.polylines(img, [np.array(roi, np.int32)], True, col, 1)
            cv2.putText(img, name, roi[0], cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    for o in world.objects:
        x, y, bw, bh = o.box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
        cv2.putText(img, o.label(), (x, max(8, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
    return img
