"""The sim backend: one MetalSim relay table, a scripted motion per skill, and a verifier per skill.

Every action is "do it, then measure it": the motion comes from sim/demos.py's primitives (top-down grasps about
Dream's tool point, IK, closed-loop plug servoing), and success is decided only by the scene's state afterwards, the
way the station backend decides it from the overhead camera. So the agent sees real failures (a missed grasp, a
tower that tips) and has to recover, as it will on the station.

All MuJoCo access goes through one lock: DimOS may call skills from several MCP request threads.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time
from typing import Any

import numpy as np

SIM_DIR = pathlib.Path(__file__).resolve().parent.parent / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from demos import TIP_CLEAR_Z, TOP_CUBE_OFFSET, TOP_DOWN, Demo, DemoFailed  # noqa: E402
from metal_sim import (  # noqa: E402
    BALL_NAMES, BALL_R, CUBE_HALF_Z, FPS, PASS_HEIGHT_M, REST_Q, ROD_HALF_LEN, MetalSim, Pose,
)

from metal_sim import CAMERAS as SIM_CAMERAS  # noqa: E402
from dimos_hillclimb.rules import EIGHT_BALL, MatchRules, Outcome, Refused  # noqa: E402

CUBES = ("blue", "green", "purple")
RODS = ("red", "green", "yellow")
HOLD_S = 3.0
CLEAR_OF_EIGHT_M = 0.06
CLEAR_OF_WHITE_M = 0.04


class _RenderThread:
    """All OpenGL on one thread. MuJoCo's renderer holds a GL context that deadlocks (CGLLockContext on macOS) when
    two threads use it, and DimOS calls skills, and the dashboard asks for frames, from different threads."""

    def __init__(self, sim: MetalSim) -> None:
        import queue
        self.sim = sim
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="sim-render", daemon=True).start()

    def _run(self) -> None:
        import mujoco
        renderer = mujoco.Renderer(self.sim.model, self.sim.height, self.sim.width)
        while True:
            camera, done = self.jobs.get()
            try:
                renderer.update_scene(self.sim.data, camera=camera)
                done.set_result(renderer.render().copy())
            except BaseException as exc:  # noqa: BLE001
                done.set_exception(exc)

    def render(self, camera: str, timeout_s: float = 10.0) -> np.ndarray:
        from concurrent.futures import Future
        done: Future = Future()
        self.jobs.put((camera, done))
        return done.result(timeout=timeout_s)


class SimStation:
    def __init__(self, rules: MatchRules, seed: int | None = None) -> None:
        self.rules = rules
        self.seed = seed
        self.sim = MetalSim(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.demo = Demo(self.sim, self._tick, self.rng)
        self.lock = threading.RLock()
        self.live_every = 6  # render an overhead frame every N ticks during motions (0: off)
        self.latest_rgb: np.ndarray | None = None
        self.latest_cams: dict[str, np.ndarray] = {}  # role -> RGB, for the dashboard's camera section
        self.cams_wanted_until = 0.0  # the wrist cameras render during motions only while someone is watching
        self._renderer: _RenderThread | None = None
        self.ticks = 0
        self.tower: list[str] = []  # bodies, base first, as the station believes it built them
        self.sim.reset("relay")

    def _tick(self) -> None:
        self.ticks += 1
        if self.live_every and self.ticks % self.live_every == 0:
            self.latest_rgb = self._render_overhead()  # for the dashboard while a motion holds the lock
            self.latest_cams["overhead"] = self.latest_rgb
            if time.monotonic() < self.cams_wanted_until:
                for role in ("wrist_l", "wrist_r"):
                    self.latest_cams[role] = self._renderer.render(SIM_CAMERAS[role])

    def _render_overhead(self) -> np.ndarray:
        """Call with self.lock held: the render thread reads sim.data while this thread waits."""
        if self._renderer is None:
            self._renderer = _RenderThread(self.sim)
        return self._renderer.render("overhead")

    # ------------------------------------------------------------------ match
    def start(self, own_color: str | None = None, seed: int | None = None) -> Outcome:
        with self.lock:
            if own_color:
                if own_color not in self.rules.ball_colors:
                    raise Refused(f"own_color must be one of {self.rules.ball_colors}")
                self.rules.own_color = own_color
            if seed is not None:
                self.sim.rng = np.random.default_rng(seed)
            self.sim.reset("relay")
            self.ticks, self.tower = 0, []
            self.rules.start()
            return Outcome(True, f"match started; own colour {self.rules.own_color}", scene=self.scene())

    # ------------------------------------------------------------------ perception
    def scene(self) -> dict:
        """The world as the agent may know it: object positions (m, table frame), the tower, the bin."""
        with self.lock:
            sim = self.sim
            def xy(body: str) -> list[float]:
                p = sim.body_pose(body).p
                return [round(float(p[0]), 3), round(float(p[1]), 3)]
            balls = sim.balls()
            return {
                "usb_cable": {"plugged_in": not sim.usb_unplugged(), "plug_xy": xy("usb_plug")},
                "cubes": {c: {"xy": xy(f"cube_{c}"), "on_tower": f"cube_{c}" in self._tower_bodies()} for c in CUBES},
                "rods": {r: {"xy": xy(f"rod_{r}"), "on_tower": f"rod_{r}" in self._tower_bodies(),
                             "upright": bool(abs(sim.body_pose(f"rod_{r}").R[2, 2]) > 0.95)} for r in RODS},
                "tower_spot_xy": [round(float(v), 3) for v in sim.tower_xy],
                "tower": self._tower_report(),
                "basket_xy": xy("basket"),
                "balls": {n: b for n, b in balls.items() if b["on_table"]},
                "match": self.rules.summary(),
            }

    def overhead_rgb(self) -> np.ndarray:
        with self.lock:
            self.latest_rgb = self._render_overhead()
            return self.latest_rgb

    def camera_names(self) -> list[str]:
        return list(SIM_CAMERAS)

    def camera_rgb(self, role: str) -> np.ndarray | None:
        """One camera's frame; while a motion holds the sim, the newest one rendered during it (or None)."""
        self.cams_wanted_until = time.monotonic() + 3.0
        if not self.lock.acquire(timeout=0.05):
            return self.latest_cams.get(role)
        try:
            if self._renderer is None:
                self._renderer = _RenderThread(self.sim)
            self.latest_cams[role] = self._renderer.render(SIM_CAMERAS[role])
            return self.latest_cams[role]
        finally:
            self.lock.release()

    # ------------------------------------------------------------------ tower measurement
    def _tower_bodies(self) -> list[str]:
        """Pieces standing stacked on the tower spot, bottom first (measured, not remembered)."""
        return self.sim.tower_pieces()

    def _tower_report(self) -> dict:
        bodies = self._tower_bodies()
        return {"pieces": [b.replace("_", " ") for b in bodies], "height_m": round(self.sim.tower_height(), 3),
                "pass_height_m": PASS_HEIGHT_M}

    # ------------------------------------------------------------------ motion helpers
    def _run(self, what: str, fn) -> Outcome | None:
        """Runs a scripted motion; a DemoFailed becomes a failed Outcome (the arms go back to rest)."""
        try:
            fn()
        except DemoFailed as exc:
            self.rules.note(f"{what}: motion failed ({exc})")
            self._rest_all()
            return Outcome(False, f"{what} failed during the motion: {exc}")
        return None

    def _rest_all(self) -> None:
        for side in ("left", "right"):
            self.sim.target_grip[side] = 1.0
        for side in ("left", "right"):
            self.demo.move_joints(side, REST_Q)
        self.demo.tick(15)

    # ------------------------------------------------------------------ stage 1
    def unplug_cable(self) -> Outcome:
        with self.lock:
            self.rules.require("unplug")
            if self.sim.usb_unplugged():
                self.rules.pass_stage("unplug", "the cable is already out")
                return Outcome(True, "the cable is already unplugged")

            def motion() -> None:
                d, sim = self.demo, self.sim
                plug = sim.body_pose("usb_plug")
                side = sim.nearest_arm(plug.p[:2])
                axis, across = plug.R[:, 0], plug.R[:, 1]
                housing = plug.p - axis * 0.019
                grasp_p = np.array([housing[0], housing[1], TIP_CLEAR_Z])
                g = d.pick(side, [Pose(grasp_p, sim.gripper_rotation(TOP_DOWN, c)) for c in (across, -across)],
                           lift=0.0)
                out = Pose(g.p - axis * 0.05, g.R)
                d.move_line(side, out, speed=0.04)
                d.move_line(side, Pose(out.p + [0, 0, 0.05], g.R), strict=False)
                d.move_line(side, Pose(out.p - axis * 0.04 + [0, 0, 0.02], g.R), strict=False)
                d.set_grip(side, 1.0)
                d.move_line(side, Pose(sim.tcp_pose(side).p + [0, 0, 0.06], g.R), strict=False)
                d.rest(side)
                d.tick(15)

            failed = self._run("unplug", motion)
            if failed:
                return failed
            if self.sim.usb_unplugged():
                self.rules.pass_stage("unplug", "a cable is out")
                return Outcome(True, "the USB cable is out of the hub; stage unplug passed (+10)",
                               match=self.rules.summary())
            return Outcome(False, "the motion finished but the plug is still in the socket")

    # ------------------------------------------------------------------ stage 2
    def insert_rod(self, rod: str, cube: str) -> Outcome:
        with self.lock:
            self.rules.require("stack")
            self._check_names(rod=rod, cube=cube)
            rod_b, cube_b = f"rod_{rod}", f"cube_{cube}"

            def motion() -> None:
                self.demo.rod_in("right", rod_b, cube_b)
                self.demo.rest("right")
                self.demo.tick(15)

            failed = self._run(f"insert the {rod} rod", motion)
            if failed:
                return failed
            rod_p, cube_p = self.sim.body_pose(rod_b), self.sim.body_pose(cube_b)
            seated = (np.linalg.norm(rod_p.p[:2] - cube_p.p[:2]) < 0.012 and abs(rod_p.R[2, 2]) > 0.95
                      and rod_p.p[2] - ROD_HALF_LEN < cube_p.p[2] + CUBE_HALF_Z)
            if seated:
                return Outcome(True, f"the {rod} rod stands in the {cube} cube's hole", tower=self._tower_report())
            return Outcome(False, f"the {rod} rod is not standing in the {cube} cube (check the scene)",
                           tower=self._tower_report())

    def stack_cube(self, cube: str, on: str) -> Outcome:
        """`on`: "tower_spot" for the base cube, or the colour of the cube to set it on (the tower's top cube)."""
        with self.lock:
            self.rules.require("stack")
            self._check_names(cube=cube)
            if on.startswith("rod:"):
                raise Refused("the tower is cubes on cubes with one rod standing in the top cube: use "
                              "stack_cube(cube, <the cube on top>) and then insert_rod(rod, <top cube>)")
            cube_b = f"cube_{cube}"
            below = None
            if on != "tower_spot":
                self._check_names(cube=on)
                below = f"cube_{on}"
            # Each cube goes a few mm off the one below, alternating sides: with every hole lined up the rod would drop
            # straight through the stack (see Demo.cube_on).
            n_cubes = sum(b.startswith("cube") for b in self._tower_bodies())
            offset = 0.0 if below is None else TOP_CUBE_OFFSET * (1 if n_cubes % 2 else -1)

            def motion() -> None:
                self.demo.cube_on("right", cube_b, below, offset=offset)
                self.demo.rest("right")
                self.demo.tick(15)

            failed = self._run(f"stack the {cube} cube", motion)
            if failed:
                return failed
            tower = self._tower_report()
            if cube_b in self._tower_bodies():
                return Outcome(True, f"the {cube} cube is on the tower; tower {tower['height_m']:.3f} m",
                               tower=tower)
            return Outcome(False, f"the {cube} cube is not standing on the tower", tower=tower)

    def check_tower(self) -> Outcome:
        """Parks both arms, waits the rubric's 3 s, and passes the stage if the tower is tall and still standing."""
        with self.lock:
            self.rules.require("stack")
            self._rest_all()
            before = self._tower_bodies()
            self.demo.tick(int(HOLD_S * FPS))
            after = self._tower_bodies()
            tower = self._tower_report()
            if after != before:
                return Outcome(False, "the tower changed during the 3 s hold (it fell or slid)", tower=tower)
            if tower["height_m"] < PASS_HEIGHT_M:
                return Outcome(False, f"the tower is {tower['height_m']:.3f} m; the bar is {PASS_HEIGHT_M} m. "
                               "Add pieces.", tower=tower)
            self.rules.pass_stage("stack", f"tower {tower['height_m']:.3f} m standing {HOLD_S:.0f} s")
            return Outcome(True, f"stack passed (+20): tower {tower['height_m']:.3f} m standing on its own",
                           tower=tower, match=self.rules.summary())

    # ------------------------------------------------------------------ stage 3
    def ball_to_bin(self, color: str) -> Outcome:
        with self.lock:
            self.rules.require("balls", color=color)
            if color == EIGHT_BALL:
                raise Refused("the black 8-ball in our bin loses the match; never pick it")
            balls = self.sim.balls()
            avoid = [np.array(b["xy"]) for b in balls.values() if b["color"] in ("white", EIGHT_BALL)
                     and b["on_table"]]
            eight = [np.array(b["xy"]) for b in balls.values() if b["color"] == EIGHT_BALL and b["on_table"]]
            cands = []
            for name, b in balls.items():
                if b["color"] != color or b["in_basket"] or not b["on_table"]:
                    continue
                xy = np.array(b["xy"])
                if any(np.linalg.norm(xy - e) < CLEAR_OF_EIGHT_M for e in eight):
                    continue
                if any(np.linalg.norm(xy - a) < CLEAR_OF_WHITE_M for a in avoid):
                    continue
                cands.append(name)
            if not cands:
                return Outcome(False, f"no {color} ball on the table is clear of the black and white balls")
            name = cands[0]
            before = self._bin_counts()

            def motion() -> None:
                d, sim = self.demo, self.sim
                p = sim.body_pose(name).p
                side = sim.nearest_arm(p[:2])
                z = max(p[2], TIP_CLEAR_Z)
                d.pick(side, [Pose(np.array([p[0], p[1], z]), sim.gripper_rotation(TOP_DOWN, [np.cos(a), np.sin(a), 0]))
                              for a in np.linspace(0, np.pi, 6, endpoint=False)], lift=0.10)
                bxy = sim.body_pose("basket").p[:2]
                hold = sim.tcp_pose(side)
                d.go(side, [Pose(np.array([bxy[0], bxy[1], 0.17]), hold.R)])
                d.set_grip(side, 1.0, ticks=10)
                d.tick(20)
                d.rest(side)
                d.tick(15)

            failed = self._run(f"put a {color} ball in the bin", motion)
            if failed:
                return failed
            return self._after_balls(before, f"{color} ball ({name})")

    def take_out_of_bin(self, color: str) -> Outcome:
        """Removes a ball that must not be in our bin (the other colour, a white or the black one)."""
        with self.lock:
            self.rules.require("balls")
            if color == self.rules.own_color:
                raise Refused("that is our colour; leave it in the bin")
            inside = [n for n, b in self.sim.balls().items() if b["color"] == color and b["in_basket"]]
            if not inside:
                return Outcome(True, f"no {color} ball is in the bin")
            name = inside[0]
            before = self._bin_counts()

            def motion() -> None:
                d, sim = self.demo, self.sim
                p = sim.body_pose(name).p
                side = sim.nearest_arm(p[:2])
                z = max(p[2], TIP_CLEAR_Z + 0.006)
                d.pick(side, [Pose(np.array([p[0], p[1], z]), sim.gripper_rotation(TOP_DOWN, [np.cos(a), np.sin(a), 0]))
                              for a in np.linspace(0, np.pi, 6, endpoint=False)], lift=0.12)
                drop = Pose(sim.tcp_pose(side).p + [0.12, 0, -0.05], sim.tcp_pose(side).R)
                d.go(side, [drop])
                d.set_grip(side, 1.0)
                d.rest(side)
                d.tick(15)

            failed = self._run(f"take the {color} ball out", motion)
            if failed:
                return failed
            return self._after_balls(before, f"took out {color} ({name})")

    def _bin_counts(self) -> tuple[int, int, bool]:
        balls = self.sim.balls()
        own = sum(b["in_basket"] and b["color"] == self.rules.own_color for b in balls.values())
        other = sum(b["in_basket"] and b["color"] == self.rules.other_color for b in balls.values())
        eight = any(b["in_basket"] and b["color"] == EIGHT_BALL for b in balls.values())
        return own, other, eight

    def _after_balls(self, before: tuple[int, int, bool], what: str) -> Outcome:
        after = self._bin_counts()
        self.rules.set_bin(*after)
        self.rules.note(f"{what}: bin {after[0]} own, {after[1]} other, eight ball {after[2]}")
        changed = after != before
        msg = f"{what}: bin now {after[0]} own, {after[1]} other" + (", BLACK 8-BALL IN BIN" if after[2] else "")
        return Outcome(changed, msg if changed else f"{what}: the bin did not change ({msg})",
                       match=self.rules.summary())

    def park(self) -> Outcome:
        with self.lock:
            self._rest_all()
            return Outcome(True, "both arms at rest, grippers open")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check_names(rod: str | None = None, cube: str | None = None) -> None:
        if rod is not None and rod not in RODS:
            raise Refused(f"no {rod!r} rod; rods are {RODS}")
        if cube is not None and cube not in CUBES:
            raise Refused(f"no {cube!r} cube; cubes are {CUBES}")


__all__ = ["SimStation", "Outcome", "BALL_NAMES", "BALL_R"]
