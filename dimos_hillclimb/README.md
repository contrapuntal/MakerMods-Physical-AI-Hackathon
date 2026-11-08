# Hill Climb relay on Dimensional OS

DimOS as a second harness layer: the relay's sub-tasks are DimOS `@skill`s, DimOS's MCP server exposes them as tools,
and DimOS's built-in agent (an LLM, via `McpClient`) breaks the match down and sequences the calls. Our own harness
pieces stay underneath: the referee rules, the per-step verifiers, and (on the station) Dream-1 doing the motion.

```
 DimOS agent (McpClient, LLM)  ── or ──  drive.py (scripted planner)  ── or ──  Claude Code / any MCP client
              │  MCP tools over http://127.0.0.1:9990/mcp
 DimOS McpServer
              │
 HillClimbStation (DimOS Module, skills.py)    each skill: referee check -> motion -> measure -> OK / FAILED / REFUSED
              │
 rules.MatchRules (referee: stage order, never-grab white, 8-ball, score)
              │
 backend: sim  -> SimStation: MuJoCo relay table (sim/), scripted motions, ground-truth verifiers
          station -> DreamStation ──HTTP──> harness serve (Dream's Python): Dream-1 ControlLoop + vision checks
```

## The task breakdown (what the agent sees as tools)

| Stage | Skill | Passes when |
|---|---|---|
| 1. Unplug (+10) | `unplug_cable()` | one USB cable is out of the hub |
| 2. Stack (+20) | `stack_cube(cube, on)` with `on` = `tower_spot`, `rod:<colour>` or a cube colour; `insert_rod(rod, cube)`; `check_tower()` | tower at least 6 in (0.152 m) and standing 3 s after the arms park |
| 3. Balls (+5 each) | `ball_to_bin(color)`, `take_out_of_bin(color)` | per ball: +5 own colour, -5 other; the black 8-ball in the bin loses |
| any | `start_match(own_color, seed)`, `look()`, `camera()`, `match_status()`, `park_arms()` | (no motion, except `park_arms`) |

The referee refuses, in code, anything the rubric forbids: a later stage's objects before the current stage passes,
white balls and the black 8-ball. The LLM can't cause a disqualification by mistake. The suggested plan (base cube,
rod in it, cube on the rod, then check, then own-colour balls one by one) and the recovery rules are in
[prompt.py](prompt.py).

## Run it

From the workspace root (DimOS's venv has MuJoCo and the agent stack):

```bash
# skills + MCP server, no LLM needed
repos/dimos/.venv/bin/python -m dimos_hillclimb serve --seed 1
# then, in a second terminal: play a match with the scripted planner, or call tools by hand
repos/dimos/.venv/bin/python -m dimos_hillclimb drive --seed 4
repos/dimos/.venv/bin/dimos mcp call look
```

```bash
# skills + MCP server + DimOS's LLM agent, which plans the match itself
ANTHROPIC_API_KEY=... repos/dimos/.venv/bin/python -m dimos_hillclimb agent --model anthropic:claude-opus-5-5 --color orange
# send it the go (from a second terminal, through the MCP server's agent_send tool)
repos/dimos/.venv/bin/dimos mcp call agent_send --arg "message=Go. Our colour is orange."
```

`--model` takes any LangChain id (`gpt-5.6-luna` with `OPENAI_API_KEY` is DimOS's default). `langchain-anthropic`
1.4.3 is installed in `repos/dimos/.venv`, pinned to DimOS's existing `langchain-core` 1.3.3.

These runs don't claim DimOS's bus-wide Coordinator name (`serve_coordinator_rpc` off), so they work next to another
DimOS run such as `dual-makermods-metal-planner-coordinator`. That also means `dimos status` and `dimos stop` don't
see them: stop them with Ctrl-C.

## Match dashboard

Every `serve` or `agent` run also serves a dashboard at http://127.0.0.1:8770/ (`--dashboard-port 0` turns it off,
`--dashboard-host 0.0.0.0` opens it to the LAN). It shows the same page for the sim and the real station:

- the live overhead frame (on the station, with the station server's detections drawn on)
- the match clock and score, each stage's state and pass time, the tower's height against the 6 in bar, the bin
- every skill call as it happens: the arguments, OK / FAILED / REFUSED, the message, how long it took
- buttons: start a match (colour, and a layout seed in sim), park the arms, and a box that sends text to DimOS's
  agent (the go, in `agent` mode)

It runs inside the skills module (dashboard.py and dashboard.html, standard library only), because that's where the
referee and the backend both live.

## On the real station

Two processes in two Pythons. The station server holds Dream-1's control loop and the cameras (harness/station.py,
harness/station_server.py); DimOS is its client, with the referee in front of every call, so nothing reaches the
arms that the rubric forbids. The server sends Dream-1 back to "park" after every step, and Ctrl-C on it homes the
arms and switches them off, whatever DimOS is doing.

```bash
# 1. on the station, in ~/team-harness (Dream's Python); --dry-run first: Dream-1 plans, nothing moves
~/dream-env/bin/python -m harness serve --dry-run --color orange
# 2. DimOS, on the station or a laptop (then serve with --host 0.0.0.0 and pass --station-url http://<station>:8765)
repos/dimos/.venv/bin/python -m dimos_hillclimb serve --backend station
# 3. play: scripted planner, or the LLM agent (agent --backend station ...), or dimos mcp call ...
repos/dimos/.venv/bin/python -m dimos_hillclimb drive --station --color orange
```

On the station the tower is built the relay's way: `stack_cube(<darkest cube>, "tower_spot")` marks the base where
it lies, `insert_rod` puts the rod in the top cube while it's still on the table, then `stack_cube(cube, <cube below>)`
twice and `check_tower`. The agent gets this plan from `prompt.STATION_PROMPT`. Each skill is one Dream-1
instruction in relay.py's wording, verified from the overhead camera as relay.py does (3 passing polls after the
gripper opens, 25 s timeout).

### Voice control hooks for `harness run`

`harness run` can be driven from outside (for example a voice front end) with signals and a state file:

- `SIGUSR1` freezes the arms in place: they hold the measured pose, with no home move and no switch-off. It works even
  before the heavy imports finish; a freeze before the arms open aborts the run.
- `SIGINT`, `SIGTERM` and `SIGHUP` end the run: home, then park (switch off). After a freeze, an end request homes from
  the hold; a freeze during that home holds again. Once the park has started it can't be interrupted.
- `<run_dir>/state.json` (written atomically) has `version`, `state`, `pid`, `started_at`, `stage`, `dry_run`,
  `freeze_supported`, `score`, `alarm`, `error`, `updated_at`. `state` is one of `starting`, `running`, `frozen`,
  `homing`, `parking`, `done`, `error`. The symlink `~/.dreamlabs/current-harness-run` points at the run dir
  (`~/hackathon_runs/<time>` unless `--out`).
- `~/.dreamlabs/arms.lock` (holder `harness run`) keeps `dreamcli robot run` and `harness run` off the arms at once.
- Dream's `DebugView` (what the model sees) is served on :8093 during a run.
- `--color` is optional: omit it and stage 3 bins every ball except the avoid (white) balls; the 8-ball rescue is unchanged.

Offline, a fake station stands in for Dream-1 (it moves symbolic objects in answer to each instruction):

```bash
repos/dimos/.venv/bin/python -m pytest harness/test_station_server.py -q    # server, HTTP, DimOS client, referee
repos/dimos/.venv/bin/python -m harness.test_station_server --port 8765     # serve the fake, then steps 2 and 3
```

## Verified (Nov 7, station backend, no robot)

- `harness/test_station_server.py`: 5 tests pass (steps in order, the rod carried with its cube, park after every
  step, time-outs, never white or black, a ball next to the 8-ball skipped, the DimOS client's full match and its
  refusals); the 6 relay tests still pass.
- Real processes against the fake station: `harness.test_station_server` -> `dimos_hillclimb serve --backend
  station` -> `drive --station` over MCP scored 40, and the `camera` tool returned the frame.
- `harness/station.py` checked against dreamlabs 1.0.23's `robots` API (names and ControlLoop arguments) in the
  portal-makermods venv, and `harness serve --help` runs there. Not yet run on the station itself: start with
  `--dry-run`.

## Verified (Nov 7, sim backend)

- `drive` over MCP, seeds 1 to 10: 8 of 10 passed unplug and stack. Seeds 1, 3, 4, 5 and 6 scored the full 40.
  Seeds 7, 8 and 9 scored 35 (one ball missed). Seeds 2 and 10 scored 10 after the tower fell or its pieces were
  knocked out of reach. Seeds 2, 7, 8 and 9 ran before the spare-piece fallback was added to `drive.py`.
- The referee's refusals work: a ball before stack passes, white balls, the black ball.
- The agent blueprint starts, `McpClient` discovers all 14 tools, takes the go, and calls the model. The only test
  key was a placeholder, so no LLM-planned match has run yet.

## Sim versus the station

- Sim motions are scripted with ground-truth poses; on the station Dream-1 does the motion from camera images, and the
  verifiers come from the overhead camera (`harness/vision.py`). The skill surface stays the same.
- The balls are 40 mm, like the station's ping-pong balls in `harness/calibration.yaml`, but heavier so the sim
  contact stays stable.
- The match clock is wall time. Sim motions run faster than real time (about 2 to 3 s per skill), so LLM thinking
  time takes up more of the 300 s than it would on the station.
