"""Scripted demonstrations for the two Hill Climb tasks, run in MetalSim.

A demo is a sequence of motions; every 30 Hz tick calls `on_tick()`, which a recorder uses to save the frame. The
arm nearest each object does the work, the other rests. Grasps are top-down about Dream's tool point; a rod lying
flat is picked near one end and tipped upright in the air before it goes into a cube's hole.

    stack: blue cube (base) <- purple cube <- green cube, then the red rod standing in the green cube's through-hole
           (resting on the purple one): 3 x 1.3 in + 2.7 in = 6.6 in, the station's tower (harness/relay.py)
    usb:   grip the plug housing, pull it out of the strip, lift it clear, put it back and push it home
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from metal_sim import CUBE_HALF_Z, REST_Q, ROD_HALF_LEN, SIDES, MetalSim, Pose

JOINT_SPEED = 1.1  # rad/s for free-space moves
CART_SPEED = 0.07  # m/s for approaches, descents and insertions
TOP_DOWN = np.array([0.0, 0.0, -1.0])
# The fingertips reach 3.2 cm past Dream's tool point (it sits mid-finger), so a top-down tool point is kept at
# least this high above the table, or the tips drag on it and the jaw cannot close.
TIP_CLEAR_Z = 0.041


class DemoFailed(RuntimeError):
    pass


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class Demo:
    def __init__(self, sim: MetalSim, on_tick: Callable[[], None], rng: np.random.Generator) -> None:
        self.sim, self.on_tick, self.rng = sim, on_tick, rng

    # ---------------------------------------------------------- primitives
    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.sim.step()
            self.on_tick()

    def move_joints(self, side: str, q_goal, speed: float = JOINT_SPEED) -> None:
        start = self.sim.target_q[side].copy()
        n = max(6, int(np.ceil(np.abs(np.asarray(q_goal) - start).max() / speed * 30)))
        for k in range(1, n + 1):
            u = k / n
            u = u * u * (3 - 2 * u)  # smooth start and stop
            self.sim.target_q[side] = start + (np.asarray(q_goal) - start) * u
            self.tick()

    def move_line(self, side: str, goal: Pose, speed: float = CART_SPEED, strict: bool = True) -> None:
        """Straight-line move of the tool point (orientation held at goal's).

        A strict move (descents, insertions) fails if the line leaves the workspace; a loose one (lifts, retreats)
        finishes in joint space instead."""
        start = self.sim.tcp_pose(side)
        n = max(6, int(np.ceil(np.linalg.norm(goal.p - start.p) / speed * 30)))
        q = self.sim.target_q[side].copy()
        for k in range(1, n + 1):
            p = start.p + (goal.p - start.p) * (k / n)
            q, err = self.sim.ik(side, Pose(p, goal.R), q0=q, iters=80)
            if err > 0.015:
                if strict:
                    raise DemoFailed(f"{side}: line move left the workspace (err {err:.3f})")
                self.go(side, [goal] + [Pose(goal.p, self.sim.tcp_pose(side).R)])
                return
            self.sim.target_q[side] = q
            self.tick()

    def set_grip(self, side: str, g: float, ticks: int = 12) -> None:
        start = self.sim.target_grip[side]
        for k in range(1, ticks + 1):
            self.sim.target_grip[side] = start + (g - start) * k / ticks
            self.tick()
        self.tick(6)

    def go(self, side: str, goals: list[Pose]) -> Pose:
        """Joint-space move to the cheapest reachable goal of equivalent options; returns it."""
        res = self.sim.reachable_goal(side, goals)
        if res is None:
            raise DemoFailed(f"{side}: no reachable pose among {len(goals)} options")
        self.move_joints(side, res[0])
        return res[1]

    def rest(self, side: str) -> None:
        self.move_joints(side, REST_Q)

    def other(self, side: str) -> str:
        return SIDES[1 - SIDES.index(side)]

    # ---------------------------------------------------------- pick and place
    def pick(self, side: str, grasps: list[Pose], lift: float = 0.10) -> Pose:
        """Approach from above, descend, close, lift; returns the object's pose in the tool frame."""
        above = [Pose(g.p + [0, 0, 0.07], g.R) for g in grasps]
        chosen = self.go(side, above)
        grasp = Pose(chosen.p - [0, 0, 0.07], chosen.R)
        self.move_line(side, grasp)
        self.set_grip(side, 0.0, ticks=14)
        self.tick(6)
        if self.sim.grip(side) < 0.04:  # the jaw closed on nothing
            raise DemoFailed(f"{side}: grasp missed")
        self.move_line(side, Pose(grasp.p + [0, 0, lift], grasp.R), strict=False)
        return grasp

    def held(self, side: str, body: str) -> Pose:
        return self.sim.tcp_pose(side).inv() @ self.sim.body_pose(body)

    def place(self, side: str, body: str, obj_goals: list[Pose], approach: float = 0.05,
              lift: float = 0.08, retreat: str = "up") -> None:
        """Move the held object to the reachable goal pose (approach from above), open, retreat up."""
        rel = self.held(side, body)
        tool_goals = [g @ rel.inv() for g in obj_goals]
        pre = self.go(side, [Pose(t.p + [0, 0, approach], t.R) for t in tool_goals])
        goal = Pose(pre.p - [0, 0, approach], pre.R)
        self.move_line(side, goal, speed=0.05)
        self.set_grip(side, 1.0, ticks=10)
        if retreat == "back":  # a sideways hand backs off along its tool axis, then rises
            back = Pose(goal.p - goal.R[:, 0] * 0.07, goal.R)
            self.move_line(side, back, strict=False)
            self.move_line(side, Pose(back.p + [0, 0, 0.04], goal.R), strict=False)
        else:
            self.move_line(side, Pose(goal.p + [0, 0, lift], goal.R), strict=False)

    # ---------------------------------------------------------- object-specific grasps
    def cube_grasps(self, body: str) -> list[Pose]:
        p = self.sim.body_pose(body)
        yaw = np.arctan2(p.R[1, 0], p.R[0, 0])
        out = []
        for k in range(4):
            a = yaw + k * np.pi / 2
            out.append(Pose(np.array([p.p[0], p.p[1], max(p.p[2] + 0.008, TIP_CLEAR_Z + p.p[2] - CUBE_HALF_Z)]), self.sim.gripper_rotation(TOP_DOWN, [np.cos(a), np.sin(a), 0])))
        return out

    def rod_grasps(self, body: str) -> list[Pose]:
        """A standing rod: top-down, the fingertips 5.5 cm below its top end (a long, steady contact)."""
        p = self.sim.body_pose(body)
        top = p.p + p.R[:, 2] * ROD_HALF_LEN * np.sign(p.R[2, 2] or 1)
        point = np.array([p.p[0], p.p[1], top[2] - 0.055 + 0.032])  # tips 5.5 cm below the top
        return [Pose(point, self.sim.gripper_rotation(TOP_DOWN, [np.cos(a), np.sin(a), 0]))
                for a in np.linspace(0, np.pi, 6, endpoint=False)]

    # ---------------------------------------------------------- the two tasks
    def cube_on(self, side: str, cube: str, below: str | None) -> None:
        """Picks `cube` and sets it flat on `below` (a cube body), or on the tower spot when `below` is None."""
        sim = self.sim
        if below is None:
            centre, base_yaw = np.array([*sim.tower_xy, CUBE_HALF_Z + 0.002]), 0.0
        else:
            b = sim.body_pose(below)
            centre = np.array([b.p[0], b.p[1], b.p[2] + 2 * CUBE_HALF_Z + 0.002])
            base_yaw = float(np.arctan2(b.R[1, 0], b.R[0, 0]))
        # Square on the cube below first (its edges line up), then any other yaw the arm can reach.
        yaws = [base_yaw + k * np.pi / 2 for k in range(4)] + [base_yaw + np.pi / 4 + k * np.pi / 2 for k in range(4)]
        self.pick(side, self.cube_grasps(cube), lift=0.06)
        self.place(side, cube, [Pose(centre, _rz(a)) for a in yaws], approach=0.04)

    def rod_in(self, side: str, rod: str, cube: str) -> None:
        """Picks a standing rod and lowers it into `cube`'s through-hole, down onto whatever is under that cube."""
        sim = self.sim
        self.pick(side, self.rod_grasps(rod), lift=0.07)
        c = sim.body_pose(cube).p
        centre_z = c[2] - CUBE_HALF_Z + ROD_HALF_LEN + 0.002  # the rod's foot on the cube's bottom face's level
        self.place(side, rod, [Pose(np.array([c[0], c[1], centre_z]), _rz(k * np.pi / 6)) for k in range(12)],
                   approach=0.04)

    def stack(self) -> None:
        """Builds the station's tower: three cubes, then a rod standing in the top cube."""
        side = "right"
        self.cube_on(side, "cube_blue", None)
        self.cube_on(side, "cube_purple", "cube_blue")
        self.cube_on(side, "cube_green", "cube_purple")
        self.rod_in(side, "rod_red", "cube_green")
        self.rest(side)
        self.tick(30)

    def usb(self) -> None:
        sim = self.sim
        plug = sim.body_pose("usb_plug")
        side = sim.nearest_arm(plug.p[:2])
        axis = plug.R[:, 0]  # the plug points along its x into the socket
        across = plug.R[:, 1]
        housing = plug.p - axis * 0.019  # the housing's back half: the fingers' front edge stays 10 mm off the socket
        grasp_p = np.array([housing[0], housing[1], TIP_CLEAR_Z])  # tips 9 mm off the table (the arm sags ~4 mm), on the housing's sides
        grasps = [Pose(grasp_p, sim.gripper_rotation(TOP_DOWN, c)) for c in (across, -across)]
        # Where the plug sits when fully home, kept relative to the strip (the strip may shift while pulling).
        seated_in_strip = sim.body_pose("strip").inv() @ sim.body_pose("usb_plug")
        g = self.pick(side, grasps, lift=0.0)
        g = Pose(g.p.copy(), g.R)
        # pull straight out, lift clear, hold
        out = Pose(g.p - axis * 0.045, g.R)
        self.move_line(side, out, speed=0.04)
        self.move_line(side, Pose(out.p + [0, 0, 0.06], g.R), strict=False)
        self.tick(int(self.rng.integers(8, 20)))
        # come back 3 cm in front of the socket, aiming the plug itself (it may have slipped in the hand)
        def plug_goal(depth: float) -> Pose:
            seated = sim.body_pose("strip") @ seated_in_strip
            return Pose(seated.p + seated.R[:, 0] * depth, seated.R)

        self.go(side, [plug_goal(-0.03) @ self.held(side, "usb_plug").inv()])
        self.tick(6)
        # line up in front of the socket, then push home, correcting from the plug's measured pose every tick
        for _ in range(4):  # hold in front of the socket until the plug is lined up within 1 mm
            if self.servo_plug(side, plug_goal, -0.03, -0.03, ticks=20) < 0.001:
                break
        self.servo_plug(side, plug_goal, -0.03, 0.003, ticks=70)
        self.tick(8)
        self.set_grip(side, 1.0, ticks=10)
        home = self.sim.tcp_pose(side)
        self.move_line(side, Pose(home.p + [0, 0, 0.08] - axis * 0.01, home.R), strict=False)
        self.rest(side)
        self.tick(30)

    def servo_plug(self, side: str, plug_goal, depth0: float, depth1: float, ticks: int) -> float:
        """Moves the held plug along the socket axis from depth0 to depth1 (m, + is into the socket), closing the
        loop on its measured pose: the tool command carries an integral correction for the arm's tracking error
        and the plug's pose in the hand. Returns the plug's final miss off the socket axis (m)."""
        sim = self.sim
        offset = np.zeros(3)
        q = sim.target_q[side].copy()
        for k in range(1, ticks + 1):
            depth = depth0 + (depth1 - depth0) * min(1.0, k / (0.8 * ticks))
            want = plug_goal(depth)
            tool_goal = want @ self.held(side, "usb_plug").inv()
            q, err = sim.ik(side, Pose(tool_goal.p + offset, tool_goal.R), q0=q, iters=40)
            if err > 0.015:
                raise DemoFailed(f"{side}: insertion pose out of reach (err {err:.3f})")
            sim.target_q[side] = q
            self.tick()
            plug_err = want.p - sim.body_pose("usb_plug").p
            offset = np.clip(offset + 0.6 * plug_err, -0.02, 0.02)
        # the plug's sideways/vertical miss from the socket axis, m
        axis = plug_goal(0.0).R[:, 0]
        return float(np.linalg.norm(plug_err - axis * (plug_err @ axis)))
