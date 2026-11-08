# Voice agent (LiveKit) for the hill climb

A LiveKit voice agent that runs on a laptop next to the station. It answers questions about the table from the
station cameras ("What can you see?", "How many balls are in the basket?", "Check the cables"). With the voice-freeze
harness it can also start, freeze and end a `harness run` by voice. This folder is a copy of the agent from the team's
working repo, so the hackathon repo holds everything the demo uses.

## Cameras

The agent never opens the station's cameras. It reads the frames Dream-1 already gets, from Dream's DebugView page,
which the harness serves on nuck-3's `127.0.0.1:8093` while a run is active (`harness/dream_executor.py`, `VIEW_PORT`).

- **How frames reach the laptop:** in harness mode each look asks nuck-3 over SSH for just the newest frame of the
  camera it needs (`CAMERA_STATION_SSH`, set by the launcher). That's about 70 KB and well under a second. A small
  standard-library script goes on stdin; nothing is installed or written on the station. Downloading the whole round
  trip through the tunnel instead is 1 MB or more, which took 5-20 s on the event Wi-Fi.
- **Turning it off:** `CAMERA_STATION_SSH=` (empty) falls back to the tunnel.

- **Frames go through privacy masking first.** The top of each frame is greyed out, then faces are blurred (YuNet),
  before anything is logged or sent to the vision model (`src/vision.py`).
- **Vision models:** `google/gemini-3.6-flash` through LiveKit Inference, with `google/gemini-3.8-flash` as the second
  try (`VISION_MODEL`, `VISION_FALLBACK_MODEL`).
- **Between runs** the page has no new frames, and the agent says so.

## Run it

```bash
cd agent
uv sync                     # Python 3.12; torch comes from the CPU index
# Put LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the environment (never in a file in the repo).

ops/run-voice.sh            # dry run (default): voice start runs `harness run --dry-run`; nothing moves
ops/run-voice.sh --check    # preflight on the station (read-only)
ops/run-voice.sh --fake     # local test against a fake harness: no station at all
ops/run-voice.sh --stop     # stop the agent (Ctrl-C in its terminal does the same)
```

- **What `ops/run-voice.sh` does:** brings the SSH tunnels up (`ops/tunnels.sh`), refuses to start a second copy of
  the agent, sets `LIVEKIT_AGENT_DEPLOYMENT=production` and the team's `HARNESS_RUN_ARGS`, then runs
  `ops/voice-agent.sh`.
- **Connecting:** in the LiveKit Agent Console, connect to `pah2026-robot-<hostname>` (or set `AGENT_NAME`).
- **Live motion** (`ops/run-voice.sh --live`) is for a person at the station only. The launcher asks for a typed confirmation, and
  someone must be at the nuck-3 keyboard to press Ctrl-C.

## Voice start, freeze and end need the voice-freeze harness

- **This branch's harness has only the camera view.** The camera questions work during any run.
- **Start, freeze and end** need the harness from branch `feat/voice-freeze-station`, installed on nuck-3 as
  `~/team-harness-voice` (the launcher's default `HARNESS_DIR`). That harness writes the run state and handles the
  freeze signal.
- **Against a harness without it,** the agent reports "another harness process is running" and doesn't signal it.

## Settings (environment)

| Variable | Default | What it does |
|---|---|---|
| `HARNESS_HOST`, `HARNESS_DIR` | `nuck-3`, `~/team-harness-voice` | where voice start runs the harness |
| `HARNESS_RUN_ARGS` | empty | extra `harness run` flags, checked against an allowlist, e.g. `--color orange --gravity-scale 0 --all-colors` |
| `DREAM_CHECKPOINT` | unset | one model for every stage; unset = the harness's per-stage fine-tunes |
| `HARNESS_TIME_LIMIT` | `3600` | match clock in seconds (the harness's own default is 300) |
| `VISION_MODEL`, `VISION_FALLBACK_MODEL` | `google/gemini-3.6-flash`, `google/gemini-3.8-flash` | camera answers; `off` disables the fallback |
| `AGENT_LLM` | LiveKit Inference | `AGENT_LLM=omlx` uses an OpenAI-compatible server instead: `OMLX_BASE_URL`, `OMLX_MODEL`, `OMLX_API_KEY` |
| `TUNNEL_HOST` | `nuck-3` | the SSH host `ops/tunnels.sh` forwards to |
| `CAMERA_STATION_SSH` | `HARNESS_HOST` in harness mode | the host to fetch frames from over SSH; empty = the tunnel |

`docs/voice-commands.md` has every voice command, every spoken line and the harness prompts per stage.
