"""Records scripted sim demonstrations into a LeRobot v2.1 dataset, with the hackathon kit's own writer and layout.

    repos/dimos/.venv/bin/python sim/record.py --task stack --episodes 20 --out ~/hackathon_data/sim-tower
    repos/dimos/.venv/bin/python sim/record.py --task usb   --episodes 20 --out ~/hackathon_data/sim-usb

Only episodes whose success check passes are kept (a failed scripted attempt is discarded, as a fumbled real demo
would be). The dataset matches what `dhk-collect` writes on a Metal station: `robot_type` metal, 30 fps, 640x360,
state/action `[left joints x6, left gripper mm, right joints x6, right gripper mm]`, and the camera keys
`observation.images.scene`, `observation.images.left_wrist`, `observation.images.right_wrist`. So the kit's
`tools/add_eef.py` and `tools/finetune.py --check` take it unchanged.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "repos/physical-ai-hack-2026/src"))

from dream_hackathon import dataset, layouts  # noqa: E402

from demos import Demo, DemoFailed  # noqa: E402
from metal_sim import FPS, MetalSim  # noqa: E402

TASK_TEXT = {
    "stack": "stack the blue, purple and green cubes into a tower and stand a rod in the top cube",
    "usb": "unplug the USB cable from the power strip and plug it back in",
}


def state_names() -> list[str]:
    names = []
    for side in layouts.SIDE_ORDER["makermods"]:
        names += [f"{side}_joint_{i}" for i in range(layouts.JOINTS)] + [f"{side}_gripper"]
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=sorted(TASK_TEXT), required=True)
    ap.add_argument("--episodes", type=int, default=10, help="successful episodes to keep")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="dataset folder (created or appended to)")
    ap.add_argument("--seed", type=int, default=0, help="first scene seed")
    ap.add_argument("--max-attempts", type=int, default=0, help="give up after this many attempts (0: 6x episodes)")
    args = ap.parse_args()
    keys = layouts.video_keys("makermods")  # role -> dataset video key
    writer = dataset.DatasetWriter(args.out.expanduser(), FPS, "metal", state_names(), list(keys.values()), 640, 360)
    kept, attempt, seed = 0, 0, args.seed
    limit = args.max_attempts or 6 * args.episodes
    t0 = time.time()
    while kept < args.episodes and attempt < limit:
        attempt += 1
        sim = MetalSim(seed=seed)
        scene_task = "usb" if args.task == "usb" else "stack"
        sim.reset(scene_task)
        ep = writer.start_episode(TASK_TEXT[args.task])
        unplugged = [False]

        def on_tick() -> None:
            images = sim.images()
            ep.add_frame(sim.state(), sim.action(),
                         {keys[role]: np.ascontiguousarray(img[:, :, ::-1]) for role, img in images.items()})
            if scene_task == "usb" and sim.usb_unplugged():
                unplugged[0] = True

        demo = Demo(sim, on_tick, np.random.default_rng(seed))
        reason = ""
        try:
            if args.task == "usb":
                demo.usb()
                ok = unplugged[0] and sim.usb_seated()
            else:
                demo.stack()
                ok = sim.tower_passes() and any(p.startswith("rod") for p in sim.tower_pieces())
        except DemoFailed as e:
            ok, reason = False, str(e)
        if ok:
            index = ep.save()
            kept += 1
            print(f"seed {seed}: kept as episode {index} ({ep.frames} frames, {ep.frames / FPS:.1f} s)", flush=True)
        else:
            ep.discard()
            print(f"seed {seed}: discarded {reason or '(task check failed)'}", flush=True)
        seed += 1
    print(f"done: {kept} kept of {attempt} attempts in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
