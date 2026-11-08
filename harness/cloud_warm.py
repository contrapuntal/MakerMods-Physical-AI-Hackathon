"""Warms Dream fine-tunes on Dream's servers without the arms or cameras: replays one saved observation.

A fine-tune starts on Dream's servers only when it is asked for actions, and its first requests are refused
`pool_warming` until it is up (a cold one took minutes in practice). This sends each model the camera frames and
arm state Dream saved from the station's last run (~/.dreamlabs/last-run_inputs: what the loop really handed it),
with the model's own task sentence, and asks again until every model answers. All models are asked from their own
thread, so they boot together. The answers are thrown away: nothing is sent to the arms, which stay free for a run.

    ~/dream-env/bin/python -m harness.cloud_warm                 # the team's five fine-tunes
    ~/dream-env/bin/python -m harness.cloud_warm MODEL [MODEL ...]
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys
import threading
import time

INPUTS = pathlib.Path("~/.dreamlabs/last-run_inputs").expanduser()
STATION = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
LIMIT_S = 20 * 60


def saved_observation() -> tuple[dict[str, str], dict]:
    """The first saved tick of the last run: frames as base64 JPEGs under `video.<camera>`, and its state."""
    for line in (INPUTS / "inputs.jsonl").open():
        row = json.loads(line)
        if row.get("kind") == "inputs":
            frames = {f"video.{cam}": base64.b64encode((INPUTS / meta["file"]).read_bytes()).decode()
                      for cam, meta in row["frames"].items()}
            return frames, dict(row["state"])
    raise SystemExit(f"no saved inputs in {INPUTS}: do one run on the arms first")


def warm(model: str, text: str, frames: dict[str, str], state: dict, out: dict) -> None:
    from robots import config as rconfig, run as rrun

    t0 = time.monotonic()
    try:
        app = rrun.with_task(rconfig.load(STATION), text)
        client = rrun.open_client(rrun.policy_config(app, model))
        if hasattr(client, "wait_for_ik"):
            client.wait_for_ik(getattr(rrun, "TRAJ_WARMUP_S", 60.0))
        obs = dict(state, prompt=text)
        tries = 0
        while time.monotonic() - t0 < LIMIT_S:
            tries += 1
            client.set_observation(frames, obs, timestamp=time.monotonic())
            try:
                t = time.monotonic()
                client.get_action(commanded=None)
                out[model] = f"WARM: answered in {time.monotonic() - t:.1f} s ({time.monotonic() - t0:.0f} s, {tries} tries)"
                break
            except Exception as exc:  # noqa: BLE001 - pool_warming is a retryable DreamServerError
                why = getattr(exc, "reason", None) or type(exc).__name__
                wait = float(getattr(exc, "retry_after_s", None) or 10.0)
                out[model] = f"booting ({why}, {time.monotonic() - t0:.0f} s)"
                print(f"[{time.strftime('%H:%M:%S')}] {model}: {why}; asking again in {wait:.0f} s", flush=True)
                if why not in ("pool_warming", "rate_limited", "pool_full") and "warming" not in str(exc):
                    out[model] = f"FAILED: {exc}"
                    break
                time.sleep(min(wait, 30.0))
        with_reset = getattr(client, "reset_episode", None)
        if with_reset:
            with_reset()
        client.close()
    except Exception as exc:  # noqa: BLE001
        out[model] = f"FAILED: {exc}"
    print(f"[{time.strftime('%H:%M:%S')}] {model}: {out.get(model)}", flush=True)


def main() -> None:
    from harness.__main__ import MODEL_TEXTS, TRAINED_TEXTS

    models = sys.argv[1:] or list(MODEL_TEXTS)
    frames, state = saved_observation()
    print(f"warming {len(models)} models from saved inputs (no arms, no cameras)", flush=True)
    out: dict[str, str] = {}
    threads = [threading.Thread(target=warm, args=(m, MODEL_TEXTS.get(m, TRAINED_TEXTS["unplug"]), frames, state, out),
                                daemon=True) for m in models]
    for th in threads:
        th.start()
        time.sleep(1.0)
    for th in threads:
        th.join()
    print("\nsummary:")
    for m in models:
        print(f"  {m}: {out.get(m)}")
    print("ALL WARM" if all(str(out.get(m, "")).startswith("WARM") for m in models) else "NOT ALL WARM", flush=True)


if __name__ == "__main__":
    main()
