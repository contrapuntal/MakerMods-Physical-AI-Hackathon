"""Renders the three cameras plus an outside view for a joint pose, to check the scene by eye.

    repos/dimos/.venv/bin/python sim/render_check.py OUT_DIR [j1 j2 j3 j4 j5 j6 (deg)]
"""

import pathlib
import sys

import cv2
import mujoco
import numpy as np

SCENE = pathlib.Path(__file__).resolve().parent / "assets/scene.xml"


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    q = np.radians([float(x) for x in sys.argv[2:8]]) if len(sys.argv) >= 8 else np.zeros(6)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    for side in ("left", "right"):
        for i in range(6):
            adr = m.joint(f"{side}_joint{i + 1}").qposadr[0]
            d.qpos[adr] = q[i]
            d.ctrl[m.actuator(f"{side}_act_joint{i + 1}").id] = q[i]
    mujoco.mj_forward(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, 360, 640)
    for cam in ("overhead", "left_wrist", "right_wrist"):
        r.update_scene(d, camera=cam)
        cv2.imwrite(str(out / f"{cam}.jpg"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    free = mujoco.MjvCamera()
    free.lookat[:] = [0.3, 0, 0.15]
    free.distance, free.azimuth, free.elevation = 1.6, 200, -25
    r.update_scene(d, camera=free)
    cv2.imwrite(str(out / "outside.jpg"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
    for side in ("left", "right"):
        print(side, "tcp", np.round(d.site(f"{side}_tcp").xpos, 3))


if __name__ == "__main__":
    main()
