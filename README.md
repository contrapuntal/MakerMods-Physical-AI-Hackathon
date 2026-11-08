# MakerMods Physical AI Hackathon: workspace

**When/where:** Nov 7 (setup workshop 15:00–17:00) to Nov 8, 2026 (demos and judging from 15:00).
Founders, Inc. SF Lab, 2 Marina Blvd B300, Building C, 3rd floor. Event page: https://luma.com/pah

**Provided on site:** bimanual 6-axis MakerMods arm stations, Dream-1 access, LiveKit credits.

**Tracks**
1. **Hill Climb:** collect demos, fine-tune Dream-1, and build a harness that picks the next action, checks whether it worked, and recovers. Ranked only by eval score.
2. **Creative Applications:** build an app around Dream-1 and the arms, for example driven by a LiveKit voice agent. Judged by a panel.

## Layout

| Path | What it is |
|---|---|
| `repos/portal-makermods` | LiveKit's reference loop: run Dream-1, take over from the leader arms, record corrections, train ACT. **Start here.** |
| `repos/maker-arm-sdk` | Maker Arm v1 driver and the `maker-arm` CLI (`doctor`, `scan`, `zero`, `check`, `teleop`). macOS uses a serial SLCAN adapter. |
| `repos/makermodslab` | MakerMods Lab web UI for calibrating, teleoperating, recording, training, and evaluating. **Close it before using any SDK**, because it holds the CAN adapter. |
| `repos/metal-arm-harness` | Safety-enveloped CLI that an agent (Claude Code, for example) can drive. Supports `--arm maker` and `--backend sim`, so it's a good base for the Hill Climb harness. |
| `repos/metal-python-ros` | Metal Arm SDK and ROS integration (reference). Its URDF and meshes feed the sim. |
| `repos/dimos` | Dimensional OS (source checkout; manipulation, sim and learning extras). Its env runs the sim. |
| `repos/physical-ai-hack-2026` | Dream Labs' hackathon kit, the same as `~/physical-ai-hacks` on the station. The sim uses its dataset writer. |
| `sim/` | Our MuJoCo copy of the NUCK-3 station and the scripted-demo recorder. See `sim/README.md`. |
| `sim_data/` | Sim datasets (LeRobot v2.1, kit layout). |
| `dimos_hillclimb/` | The relay as Dimensional OS skills: DimOS's MCP server and LLM agent plan and sequence the match on top of our referee and verifiers (sim backend, and the real station through `python -m harness serve`). See `dimos_hillclimb/README.md`. |

## Install

```bash
./setup.sh
```

This clones the upstream repos into `repos/` (gitignored) if they are missing, then installs each one. It needs Python 3.12 (installed) and `uv` (installed). The LiveKit CLI `lk` is already installed through Homebrew.

## Credentials (`repos/portal-makermods/.env`)

- `DREAM_API_KEY=dl_live_...`: create it at https://platform.dreamlabs.ai under API Keys. It is shown only once.
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`: from your LiveKit Cloud project (hackathon credits). Alternatively, run `lk cloud auth` and then `lk project list`.
- Arm CAN ports and camera indexes: fill these in at the station (see `repos/portal-makermods/tutorial/01-first-recording.md`). On macOS, find adapters with `ls /dev/cu.usbmodem* /dev/cu.usbserial*`.

## Dream-1 quick reference

- `pip install dreamlabs` gives `dreamcli` (`dreamcli setup`, then `dreamcli robot up`, then give a task in Live View).
- `pip install dreamlabs-client` gives `import dream; client = dream.DreamClient()`. Each call returns about 1.6 s of actions from camera images, robot state, and a task string.
- Docs: https://platform.dreamlabs.ai/api-documentation. Pricing: Dream-1 is $3/h and Dream-1-flash is $2/h.

## Still to come from organizers

Challenge and scoring rules, the creative rubric, and station booking. Grab them at the 15:00 workshop.
