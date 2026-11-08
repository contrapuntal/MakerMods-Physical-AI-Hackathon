"""The NUCK-3 Hill Climb simulation: two Metal arms, randomized props, IK about Dream's tool point, task checks.

Everything is in the kit's units (hackathon kit `dream_hackathon/layouts.py`): joints in radians, the gripper 0
(closed) to 1 (open), recorded as millimetres of jaw stroke (100 mm open on a Metal arm), vectors ordered
`[left joints x6, left gripper, right joints x6, right gripper]`.

@see MetalSim
"""

from __future__ import annotations

import dataclasses
import pathlib

import mujoco
import numpy as np
import numpy.typing as npt

SCENE = pathlib.Path(__file__).resolve().parent / "assets/scene.xml"
PORT_Z = 0.010  # must match build_scene.PORT_Z
SIDES = ("left", "right")
FPS = 30
SUBSTEPS = 16  # physics at 480 Hz under a 30 Hz control/record tick
FINGER_OPEN = -0.05  # joint7 at a fully open jaw (m); 0 is closed
GRIP_SCALE_MM = 100.0
CAMERAS = {"overhead": "overhead", "wrist_l": "left_wrist", "wrist_r": "right_wrist"}
# Tower pieces, measured at the station (defined once, in build_scene.py).
from build_scene import CUBE_HALF_XY, CUBE_HALF_Z, HOLE_HALF, ROD_HALF_LEN, ROD_R  # noqa: E402
CUBES = ("cube_blue", "cube_green", "cube_purple")
RODS = ("rod_red", "rod_green", "rod_yellow")
PASS_HEIGHT_M = 0.152  # the rubric's 6 in bar
REST_Q = np.radians([0.0, -8.0, 25.0, 15.0, 0.0, 0.0])  # a folded, resting-looking start pose
BALL_R = 0.020  # must match build_scene.BALL_R
BALL_NAMES = ("ball_orange_0", "ball_orange_1", "ball_green_0", "ball_green_1", "ball_white_0", "ball_white_1",
              "ball_black")
BASKET_IN, BASKET_H = 0.065, 0.06  # must match build_scene


def _quat_z(yaw: float) -> npt.NDArray[np.float64]:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _mat(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, quat)
    return m.reshape(3, 3)


@dataclasses.dataclass
class Pose:
    """A rigid pose: position (m) and rotation matrix, world frame."""

    p: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.p + self.R @ other.p, self.R @ other.R)

    def inv(self) -> "Pose":
        return Pose(-self.R.T @ self.p, self.R.T)


class MetalSim:
    """One simulated station. Drive it with step(joint targets), read state(), images(), and the task checks."""

    def __init__(self, seed: int | None = None, width: int = 640, height: int = 360) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.model.opt.timestep = 1.0 / (FPS * SUBSTEPS)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.width, self.height = width, height
        self._renderer: mujoco.Renderer | None = None
        m = self.model
        self.qadr = {s: [m.joint(f"{s}_joint{i}").qposadr[0] for i in range(1, 7)] for s in SIDES}
        self.dadr = {s: [m.joint(f"{s}_joint{i}").dofadr[0] for i in range(1, 7)] for s in SIDES}
        self.finger_adr = {s: m.joint(f"{s}_joint7").qposadr[0] for s in SIDES}
        self.act = {s: [m.actuator(f"{s}_act_joint{i}").id for i in range(1, 7)] for s in SIDES}
        self.grip_act = {s: m.actuator(f"{s}_act_gripper").id for s in SIDES}
        self.limits = {s: np.array([m.jnt_range[m.joint(f"{s}_joint{i}").id] for i in range(1, 7)]) for s in SIDES}
        self.tcp = {s: m.site(f"{s}_tcp").id for s in SIDES}
        self.link6 = {s: m.body(f"{s}_link6").id for s in SIDES}
        self.base_xy = {s: m.body(f"{s}_base_link").pos[:2].copy() for s in SIDES}
        self.target_q = {s: REST_Q.copy() for s in SIDES}
        self.target_grip = {s: 1.0 for s in SIDES}
        self.closing_axis = self._closing_axis()
        self.rods_upright = True

    # ---------------------------------------------------------------- scene
    def reset(self, task: str) -> None:
        """Rest pose, grippers open, props randomized for `task` ("stack" or "usb")."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for s in SIDES:
            d.qpos[self.qadr[s]] = REST_Q
            d.qpos[self.finger_adr[s]] = FINGER_OPEN
            self.target_q[s], self.target_grip[s] = REST_Q.copy(), 1.0
        r = self.rng
        # The strip: on the left half for stacking episodes (out of the way), centre-ish for USB episodes.
        if task == "usb":
            sx, sy, syaw = r.uniform(0.36, 0.42), r.uniform(0.0, 0.10), r.uniform(-0.25, 0.25)
        elif task == "relay":  # all three stages on one table: hub centre, blocks right, balls and basket left
            sx, sy, syaw = r.uniform(0.38, 0.42), r.uniform(0.0, 0.05), r.uniform(-0.15, 0.15)
        else:
            sx, sy, syaw = r.uniform(0.40, 0.46), r.uniform(0.18, 0.24), r.uniform(-0.2, 0.2)
        self._set_free("strip", [sx, sy, 0.02], _quat_z(syaw))
        strip = Pose(np.array([sx, sy, 0.02]), _mat(_quat_z(syaw)))
        plug = strip @ Pose(np.array([-0.204, 0.0, PORT_Z - 0.02]), np.eye(3))
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, plug.R.flatten())
        self._set_free("usb_plug", plug.p, q)
        # Blocks: cubes flat on the table, rods lying flat, in the right/centre region, not overlapping.
        # Keep props off the strip: block points along its length (and its plug end).
        axis = np.array([np.cos(syaw), np.sin(syaw)])
        taken: list[npt.NDArray[np.float64]] = [np.array([sx, sy]) + axis * u for u in np.arange(-0.26, 0.25, 0.06)]
        # Tower spot ~0.23 m from the right base, toward the centre line (inside the clear-of-arms rule's gap).
        tx = r.uniform(0.17, 0.21)
        ty = r.uniform(-0.17, -0.13) if task in ("stack", "relay") else 0.0
        self.tower_xy = np.array([tx, ty])
        taken.append(self.tower_xy.copy())
        # Stacking is a right-arm task (the tower stands close to its base so the top piece stays reachable).
        x_rng, y_rng = ((0.14, 0.40), (-0.46, -0.02)) if task in ("stack", "relay") else ((0.18, 0.40), (-0.42, -0.14))
        self._spawn_side = "right" if task in ("stack", "relay") else None
        if task == "relay":
            self._layout_balls(taken)
        for name in ("cube_blue", "cube_green", "cube_purple", "rod_red", "rod_green", "rod_yellow"):
            # Rods stand 4 in tall and tip when the arm brushes them: give them more room than the cubes.
            xy = self._free_spot(taken, x_rng, y_rng, clearance=0.095 if name.startswith("rod") else 0.075)
            yaw = r.uniform(-np.pi, np.pi)
            if name.startswith("cube"):
                self._set_free(name, [xy[0], xy[1], CUBE_HALF_Z], _quat_z(yaw))
            elif self.rods_upright:  # standing on end (simplification: no in-hand re-orientation needed)
                self._set_free(name, [xy[0], xy[1], ROD_HALF_LEN + 0.001], _quat_z(yaw))
            else:  # lying flat as on the real table: cylinder axis (local z) horizontal
                q_lie = np.zeros(4)
                mujoco.mju_mulQuat(q_lie, _quat_z(yaw), np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
                self._set_free(name, [xy[0], xy[1], ROD_R], q_lie)
        mujoco.mj_forward(m, d)
        self.settle(20)

    def _layout_balls(self, taken: list) -> None:
        """The relay's stage 3: the basket left of the hub, the seven balls scattered in the left arm's reach."""
        r = self.rng
        bxy = np.array([r.uniform(0.25, 0.29), r.uniform(0.20, 0.25)])
        self._set_free("basket", [bxy[0], bxy[1], 0.003], _quat_z(0.0))
        self.basket_xy = bxy
        taken.extend(bxy + np.array(c) for c in ((0, 0), (0.05, 0.05), (0.05, -0.05), (-0.05, 0.05), (-0.05, -0.05)))
        self._spawn_side = "left"
        for name in BALL_NAMES:
            xy = self._free_spot(taken, (0.14, 0.45), (0.06, 0.52), clearance=0.055)
            self._set_free(name, [xy[0], xy[1], BALL_R + 0.001], _quat_z(0.0))
        self._spawn_side = "right"

    def _free_spot(self, taken, x_rng, y_rng, clearance: float = 0.075) -> npt.NDArray[np.float64]:
        """A random spot in range, reachable, clear of the arms and `clearance` from every taken spot; on a crowded
        table the clearance is relaxed step by step (never below 6 cm) rather than giving up."""
        for c in (clearance, clearance * 0.9, clearance * 0.8, 0.06):
            c = max(c, 0.06)
            for _ in range(500):
                xy = np.array([self.rng.uniform(*x_rng), self.rng.uniform(*y_rng)])
                bases = [self.base_xy[self._spawn_side]] if self._spawn_side else list(self.base_xy.values())
                reach = min(float(np.linalg.norm(xy - b)) for b in bases)
                clear_of_arms = min(float(np.linalg.norm(xy - b)) for b in self.base_xy.values()) > 0.21
                if clear_of_arms and reach < 0.37 and all(np.linalg.norm(xy - t) > c for t in taken):
                    taken.append(xy)
                    return xy
        raise RuntimeError("no free spot for a prop")

    def _set_free(self, body: str, pos, quat) -> None:
        adr = self.model.joint(f"{body}_free").qposadr[0]
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = quat

    def body_pose(self, body: str) -> Pose:
        b = self.data.body(body)
        return Pose(b.xpos.copy(), b.xmat.reshape(3, 3).copy())

    # ---------------------------------------------------------------- control
    def step(self) -> None:
        """One 30 Hz tick toward the current targets."""
        d = self.data
        for s in SIDES:
            d.ctrl[self.act[s]] = self.target_q[s]
            d.ctrl[self.grip_act[s]] = FINGER_OPEN * float(np.clip(self.target_grip[s], 0, 1))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, d)

    def settle(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def joints(self, side: str) -> npt.NDArray[np.float64]:
        return self.data.qpos[self.qadr[side]].copy()

    def grip(self, side: str) -> float:
        return float(np.clip(self.data.qpos[self.finger_adr[side]] / FINGER_OPEN, 0, 1))

    def state(self) -> npt.NDArray[np.float32]:
        """Measured `[left joints, left gripper mm, right joints, right gripper mm]`."""
        parts = []
        for s in SIDES:
            parts += [self.joints(s), [self.grip(s) * GRIP_SCALE_MM]]
        return np.concatenate(parts).astype(np.float32)

    def action(self) -> npt.NDArray[np.float32]:
        """Commanded targets in the same layout."""
        parts = []
        for s in SIDES:
            parts += [self.target_q[s], [float(np.clip(self.target_grip[s], 0, 1)) * GRIP_SCALE_MM]]
        return np.concatenate(parts).astype(np.float32)

    def images(self) -> dict[str, npt.NDArray[np.uint8]]:
        """Per camera role (overhead, wrist_l, wrist_r), an RGB frame."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        out = {}
        for role, cam in CAMERAS.items():
            self._renderer.update_scene(self.data, camera=cam)
            out[role] = self._renderer.render().copy()
        return out

    # ---------------------------------------------------------------- kinematics
    def _closing_axis(self) -> npt.NDArray[np.float64]:
        """The jaw's closing direction in link6's frame (unit), measured from the finger bodies."""
        d = self._ik_data
        d.qpos[:] = self.model.qpos0
        for s in SIDES:
            d.qpos[self.finger_adr[s]] = FINGER_OPEN
        mujoco.mj_kinematics(self.model, d)
        s = "left"
        p7, p8 = d.body(f"{s}_link7").xpos, d.body(f"{s}_link8").xpos
        R6 = d.xmat[self.link6[s]].reshape(3, 3)
        axis = R6.T @ (p8 - p7)
        axis -= axis[0] * np.array([1.0, 0, 0])  # keep it orthogonal to the tool axis (link6 x)
        return axis / np.linalg.norm(axis)

    def gripper_rotation(self, tool_w, close_w) -> npt.NDArray[np.float64]:
        """The link6 rotation whose tool axis (x) points along tool_w and whose jaw closes along close_w."""
        t = np.asarray(tool_w, float) / np.linalg.norm(tool_w)
        c = np.asarray(close_w, float)
        c = c - t * (c @ t)
        c /= np.linalg.norm(c)
        e1, e2 = np.array([1.0, 0, 0]), self.closing_axis
        E = np.column_stack([e1, e2, np.cross(e1, e2)])
        F = np.column_stack([t, c, np.cross(t, c)])
        return F @ E.T

    def tcp_pose(self, side: str, data: mujoco.MjData | None = None) -> Pose:
        d = data or self.data
        return Pose(d.site_xpos[self.tcp[side]].copy(), d.site_xmat[self.tcp[side]].reshape(3, 3).copy())

    def ik(self, side: str, goal: Pose, q0=None, iters: int = 150) -> tuple[npt.NDArray[np.float64], float]:
        """Damped least squares to put the tool point at goal; returns (joints, residual: pos m + rot rad/10)."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        q = (self.joints(side) if q0 is None else np.asarray(q0, float)).copy()
        lo, hi = self.limits[side][:, 0], self.limits[side][:, 1]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = np.inf
        for _ in range(iters):
            d.qpos[self.qadr[side]] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cur = self.tcp_pose(side, d)
            e_p = goal.p - cur.p
            e_r = 0.5 * sum(np.cross(cur.R[:, i], goal.R[:, i]) for i in range(3))
            err = float(np.linalg.norm(e_p) + 0.1 * np.linalg.norm(e_r))
            if err < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.tcp[side])
            J = np.vstack([jp[:, self.dadr[side]], jr[:, self.dadr[side]]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        return q, err

    def reachable_goal(self, side: str, goals: list[Pose], tol: float = 0.003):
        """The goal (of equivalent options) the arm reaches with the least joint motion; None if none reaches."""
        best = None
        for g in goals:
            rel = g.p[:2] - self.base_xy[side]
            yaw = float(np.arctan2(rel[1], rel[0]))
            seeds = [self.joints(side), REST_Q] + [
                np.array([yaw, a, b, c, 0.0, 0.0])
                for a, b, c in ((-2.27, 1.99, -1.29), (-1.75, 2.44, -0.70), (-2.6, 1.75, -0.7), (-1.2, 1.6, -0.5))
            ]
            for seed in seeds:
                q, err = self.ik(side, g, q0=seed)
                if err < tol:
                    cost = float(np.abs(q - self.joints(side)).sum())
                    if best is None or cost < best[0]:
                        best = (cost, q, g)
        return None if best is None else (best[1], best[2])

    def nearest_arm(self, xy) -> str:
        return min(SIDES, key=lambda s: float(np.linalg.norm(np.asarray(xy) - self.base_xy[s])))

    # ---------------------------------------------------------------- checks (usable by a harness)
    def tower_pieces(self) -> list[str]:
        """Pieces standing stacked on the tower spot, bottom first: upright, centred over the spot, each resting on
        (or, for a rod in a cube's through-hole, inside) the one below. Measured from the scene, never remembered."""
        spot = self.tower_xy
        found = []
        for name in CUBES + RODS:
            p = self.body_pose(name)
            if np.linalg.norm(p.p[:2] - spot) < 0.03 and abs(p.R[2, 2]) > 0.95:
                found.append((float(p.p[2]), name))
        found.sort()
        out, top = [], 0.0
        for z, name in found:
            half = CUBE_HALF_Z if name.startswith("cube") else ROD_HALF_LEN
            if z - half > top + 0.005:  # a gap: not resting on what's below
                break
            out.append(name)
            top = max(top, z + half)
        return out

    def tower_height(self) -> float:
        """The top of the tower above the table (m), 0 with no tower."""
        top = 0.0
        for name in self.tower_pieces():
            half = CUBE_HALF_Z if name.startswith("cube") else ROD_HALF_LEN
            top = max(top, float(self.body_pose(name).p[2]) + half)
        return top

    def tower_passes(self) -> bool:
        return self.tower_height() >= PASS_HEIGHT_M

    def usb_seated(self) -> bool:
        """The plug's tip is inside the strip's USB socket (within 2 mm of fully seated)."""
        tip = self.data.site("plug_tip_site").xpos
        port = self.data.site("usb_port").xpos
        strip_R = self.body_pose("strip").R
        local = strip_R.T @ (tip - port)
        return abs(local[1]) < 0.003 and abs(local[2]) < 0.003 and -0.002 < local[0] < 0.012

    def usb_unplugged(self) -> bool:
        tip = self.data.site("plug_tip_site").xpos
        port = self.data.site("usb_port").xpos
        return float(np.linalg.norm(tip - port)) > 0.03

    def in_basket(self, body: str) -> bool:
        """The body's centre is inside the basket's walls and below their top edge."""
        b = self.body_pose("basket")
        local = b.R.T @ (self.body_pose(body).p - b.p)
        return bool(abs(local[0]) < BASKET_IN and abs(local[1]) < BASKET_IN and -0.01 < local[2] < BASKET_H)

    def balls(self) -> dict[str, dict]:
        """Every relay ball: colour, table position (m), whether it is in the basket or on the table."""
        out = {}
        for name in BALL_NAMES:
            p = self.body_pose(name).p
            on_table = -0.15 < p[0] < 0.95 and abs(p[1]) < 0.75 and p[2] > -0.01
            out[name] = {"color": name.split("_")[1], "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
                         "in_basket": self.in_basket(name), "on_table": bool(on_table)}
        return out
