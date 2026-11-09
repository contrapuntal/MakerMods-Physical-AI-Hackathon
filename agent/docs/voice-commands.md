# Voice agent: commands, spoken lines and robot prompts

Reference for the operator and the team. Built from the code on 2026-11-08:
- the agent in `src/`;
- the harness on branch `feat/voice-freeze-station` @ `cb91f56`, i.e. the teammates' station harness with our freeze
  merged in.

Quoted lines are exact strings from the code. Where the model phrases a reply itself, that is said.

**There is no physical e-stop on this setup.** The voice emergency word holds the arms. At the station, Ctrl-C in
the harness window ends the run.

---

## 1. What you can say (harness mode: the match path)

| Say | What happens | Needs "yes"? |
|---|---|---|
| **"Freeze"** or **"Halt"** | The arms stop and hold where they are. The run goes to `frozen`. It never homes and never parks. | No |
| **"Start the hill climb"** (also: "start the relay / run / match") | The agent reads back "start the hill climb" and asks. After your yes, the harness run starts on the station (`harness run [--checkpoint $DREAM_CHECKPOINT] $HARNESS_RUN_ARGS [--dry-run] --time-limit 3600 ...`, e.g. `HARNESS_RUN_ARGS="--color orange --lower-unplug-cm 8.5"`; see section 4). | Yes |
| **"End the run"** (also: "stop the hill climb", "end the match", "we're done", "that's it") | The agent reads back "end the run" and warns about the tower. After your yes: the home move (3 s), then park, then motors off, then `done`. | Yes |
| **"What's the status?"** (also: "how is the hill climb going", "what stage", "are the arms holding", "is it done") | It answers from the station's state. | No |
| **"No"**, **"cancel"**, **"wait"** | Cancels a pending start or end. | No |
| **"Check the tower"**, **"count the balls"**, **"look at the table"**, **"check the cables"**, **"how's it going"** | Camera answers, from the run's camera view. | No |

### The emergency word, exactly
- **The words:** `freeze` or `halt`, matched as whole words (`FREEZE_WORD_RE`, `robot_gate.py:65`).
- **Instant path:**
  - It fires on the **final** transcript only, when the word is one of the **first two** words. Examples: "Freeze!",
    "Halt the robot", "Okay, freeze", "No, freeze!".
  - It does **not** fire when the word right before it is `don't` / `not` / `never` ("Don't freeze it yet").
  - Repeats within 2 s are merged (`STOP_DEBOUNCE_S`).
- **Model backstop:**
  - Outside a match, the model also freezes on "hold still", "stop moving", "okay robot, freeze" and similar.
  - **In match mode** it freezes only if the word `freeze` or `halt` is actually in what you said. Bystander
    "hold on" or "stop" can't freeze the run.
- One utterance gives **one** freeze attempt and **one** spoken outcome: the instant path owns its turn.
- **While the agent is talking:** the agent uses LiveKit's *adaptive* interruption, which was kept on purpose. Speech
  over the agent is held until LiveKit's model decides "interruption" or "backchannel". A backchannel verdict drops the
  transcript (`audio_recognition.py` transcript gate), so **a very short "Freeze!" said over the agent could be lost.**
  If the agent is speaking, say it twice ("Freeze, freeze") or as a short phrase ("Freeze the arms"). In match mode the
  agent is mostly silent, so this window is small.

### Confirming ("yes")
- **Only short, unconditional assent counts, as the whole reply** (`ASSENT_RE`, at most 7 words): optional fillers
  (uh, um, ok, okay, so, well, alright), then yes / yeah / yep / yup / sure / go ahead / do it / go for it / confirm(ed)
  (repeats allowed), then optionally "please", "now", "thanks", "I'm sure". Accepted: "yes", "yeah", "yes please",
  "yep go ahead", "sure, do it", "okay yes".
- **Refused as qualified, conditional, quoted or reported:** "yes only when I am clear", "yes if it's safe", "yes once
  the referee says", "he said yes", "say yes", "yes after the count".
- **Restating the pending action also counts** (at most 10 words): "Yes, end the run", "Yes, start the hill climb",
  "Yes, start it", "Okay yes, send the arms home". Only the action that is pending: "Yes, start the hill climb" while
  an end is pending is refused. Any other extra words fail.
- **Never counts as a yes,** even with a yes word in it: a reply containing no / don't / not / wait / stop / cancel /
  never mind / hold on / hang on / nah / actually / abort / later / maybe / nope / cut / freeze / halt / instead /
  different / other one / wrong / after / minute / second / first (`NO_RE`). So "Okay, yeah, freeze!" freezes and does
  not confirm.
- **Timing:** it must be your **very next** message after the question, within **15 s of the agent finishing the
  question** (`PENDING_TTL_S`, restarted once when the agent stops speaking).

### After a freeze
- **"End the run"** + yes is the only way out by voice. The freeze is one-way: a run can't resume.
- **"Start the hill climb"** is refused while a run exists. End it first, then start a new run, which homes at
  start-up.
- **Saying the word again** is harmless: it re-sends the freeze.

---

## 2. What the agent says (fixed lines)

| When | Line |
|---|---|
| Freeze confirmed (instant path) | "Holding." |
| Freeze confirmed (tool) | "Sent; the arms are holding." |
| Freeze sent, still coming to a hold (the harness reports `freezing`) | "Sent; the arms are coming to a hold." (if `frozen` isn't confirmed within 8 s, it retries and reports) |
| Freeze not confirmed, retrying | "Not confirmed yet; trying again." |
| Freeze not confirmed after retries | "Not confirmed: the arms may still be moving." |
| The hold did not take: the harness says `frozen` but an arm could not hold (`hold_ok` false, or an arm alarm with no `hold_ok`) | "Not confirmed: an arm did not take the hold and may still be moving." (spoken as an alarm; never "Holding.") |
| Station unreachable | "I could not reach the station; the arms may still be moving." |
| No run, nothing holds the arms | "Nothing is running; the arms aren't moving." |
| Another program holds the arms (e.g. `dreamcli robot run`) | "Another program, …, is driving the arms; I can't stop it by voice." |
| Freeze while the run is still starting | "Accepted; the run stops before the arms move." |
| Run aborted before the arms moved | "Done; the run ended before the arms moved." |
| Freeze after the run finished | "The run is already over: the arms are parked and switched off; nothing is moving." |
| A freeze landed during an in-flight start or end | "Already handled: an emergency hold arrived while this was in flight and was re-sent afterwards; the arms are holding." |
| End accepted | "Accepted: the run is ending and the arms are going home." / "Sent: the run is ending; the arms will go home in a moment." |
| End dropped (the arms were still holding) | "the end was not taken: the arms are still holding. Say end the run again." |
| End with no run | "No run was active; nothing was sent." / "The run had already finished; nothing moved." |
| While frozen (outside match mode) | "The arms are holding. When the referee has scored, say end the run, then yes.", then "Still holding." |
| Arm alarm while frozen | "An arm raised an alarm and may drop. Please support it." |
| A second, untracked harness process | "Another harness process is running on the station; I can't stop it by voice." |
| The 8-ball lands in our bin | "The black eight ball is in our bin." |
| A run starts or moves to a new stage (read from the station view's current instruction; retries of the same stage say nothing) | "Picking up the cables now." / "Stacking the blocks now." / "Putting the golf balls into the basket now." / "Taking the black ball back out of the bin." (`STAGE_ANNOUNCE=0` turns these off) |
| "How does this work?" / "What are you?" | A two-or-three-sentence explanation from fixed facts (the arms and cameras, Dream-1 and the per-stage fine-tunes, the harness and Claude checks, the agent's models, privacy masking, safety), then an offer of one more level |

- **The end warning,** said by the model when it proposes an end: ending sends the arms home and then parks them, and
  that can knock over a standing tower. The emergency word still holds the arms during the home move, but it can't
  stop the final park.
- **The greeting** (hill climb), said word for word: "Hi there, I'm the operator for the hill climb." The launcher's banner still prints whether it is live, a dry run or a local test.

### Match mode
It is on from a confirmed start until the run ends. While it's on:
- the agent stays silent;
- it answers status questions in one short sentence;
- end and confirm still work;
- the emergency word works;
- there are no reminders.

Alarms (the 8-ball, an arm alarm, an unconfirmed freeze) are always spoken.

---

## 3. Station mode (free-form tasks; not the match path)

| Say | What happens |
|---|---|
| Any task, e.g. **"Remove Orange Cables from their port"** | The agent proposes it **word for word** (a leading label like "Object stacking:" is dropped). After your yes: Dream station `/start` with that text, or a DRY RUN that sends nothing when not live. |
| **"Go home"** | It proposes going home. After your yes: station `/home`. |
| **"Freeze"/"Halt"**, or "stop moving" | Station `/stop`: the arms hold. |

The trained task texts, word for word, read from the datasets on nuck-3:

| Fine-tune dataset | Say exactly |
|---|---|
| USB_Cable_Task | "Remove Orange Cables from their port" |
| Stacking_Boxes | "Stack the green, purple, and blue blocks" |
| Balls_In_Basket | "Place yellow, blue, green, and orange balls into basket." |
| Remove_White_Balls | "Move nearby white balls far away" |
| Place_Balls_Remove_Whites | "Place all yellow, orange, green, and blue balls in the basket. Push away all white balls." |

---

## 4. What the harness tells Dream-1 (after "start the hill climb" + yes)

Voice start launches this on nuck-3 (it mirrors the teammates' command line):
```
~/dream-env/bin/python -m harness run [--checkpoint $DREAM_CHECKPOINT] --calibration ~/team-harness/harness/calibration.yaml \
    $HARNESS_RUN_ARGS [--dry-run] --time-limit 3600 --out ~/hackathon_runs/<run>
```
- `HARNESS_RUN_ARGS` (env of `ops/voice-agent.sh`, default empty), e.g.
  `HARNESS_RUN_ARGS="--color orange --lower-unplug-cm 8.5"`. The helper checks every token against an allowlist and
  refuses the whole start on anything else: `--color WORD`, `--lower-{unplug,stack,balls}-cm 0..20`,
  `--model-{unplug,stack,balls} ID`, `--rod-on-tower`, `--all-colors`, `--stack-pass 6in|3-cubes`,
  `--gravity-scale 0..1.5`, `--augment MODES` (`[a-z,]+`), `--detector hsv|yolo|both`. Never allowed:
  `--checkpoint`, `--dry-run`, `--time-limit`, `--calibration`, `--out`, `--*-text`. `ops/voice-agent.sh --check`
  validates it too, and the banner prints it.
- **No `DREAM_CHECKPOINT` = the harness's per-stage default models** (no `--checkpoint` is sent; the launcher never
  uses the placeholder for the harness backend). Setting `DREAM_CHECKPOINT` forces ONE model for all stages.
- **Time limit:** voice start sends `--time-limit 3600` (`HARNESS_TIME_LIMIT`, the human's choice: the run does not end
  by itself). The teammates' command line uses the harness default of 300 s.
The prompts depend on the checkpoint and `--color` (`plan_stages`, `harness/__main__.py:46-61`).

### A. No checkpoint (base `dreamlabs/dream-1-beta` for all stages): step-by-step instructions
| Stage | Instruction sent (`harness/relay.py`) |
|---|---|
| unplug (rotates on retries) | "grab the orange USB cable's plug at the power strip, then pull it straight up out of the port without touching the blocks or the balls" / "pinch the orange plug where it goes into the power strip and lift it straight up until it comes out without touching the blocks or the balls" / "pull one orange USB plug straight up out of the power strip without touching the blocks or the balls" |
| stack: rod first | "put the `<rod>` rod upright in the hole of the `<cube>` cube without touching the balls or the cables" |
| stack: each cube | "place the `<cube>` cube on top of the `<cube below>` cube without touching the balls or the cables" (or "pick up the `<cube>` cube with the `<rod>` rod standing in it by the sides of the cube and place it on top of the `<cube below>` cube …") |
| stack: rod on the tower (fallback) | "put the `<rod>` rod upright in the hole of the `<cube>` cube on top of the tower without touching the balls or the cables" |
| stack: hold, park, waiting | "move both arms up and away from the tower and hold them still" |
| balls | "pick up the `<colour>` ping-pong ball and drop it in the bin and do not touch the white or the black balls" (no `--color`: the first non-white ball it sees) |
| balls: take-outs | "take the black ball out of the bin and put it on the table" / "take the `<colour>` ping-pong ball out of the bin and put it on the table" (white balls; the other colour when `--color` is set) |
| any stage, holding a leftover piece | "put the `<piece>` down on the empty table beside the power strip, away from the blocks and the balls, and open the gripper" |

### B. `DREAM_CHECKPOINT` = a fine-tune (`dream-1-ft:…`): one model for all stages, trained texts word for word
| Stage | Instruction sent |
|---|---|
| unplug | "Remove Orange Cables from their port" |
| stack | "Stack the green, purple, and blue blocks" (one whole-tower step, checked by Claude after 15 s) |
| balls | "Place yellow, blue, green, and orange balls into basket." |

The fine-tunes are per task, so one checkpoint for all three stages runs one task's model on the others. Picking a
model per stage from voice start is not built (review I-3).

---

## 5. Commands (laptop and station)

| Where | Command | What it does |
|---|---|---|
| laptop | `ops/tunnels.sh up` / `status` / `down` | SSH port forwards to nuck-3 (8080, 8091, 8093). Moves nothing. |
| laptop | `ops/voice-agent.sh --backend harness --check` | Preflight on nuck-3 (ssh, helper, help contract, arms lock, calibration). Needs the human's go-ahead. |
| laptop | `LIVEKIT_AGENT_DEPLOYMENT=production ops/voice-agent.sh --backend harness` | Starts the voice agent, dry run. Production deployment, so the Agent Console reaches it. |
| laptop | `… --backend harness --live` | Live (human only), from a real terminal (it prints the live-motion banner). |
| laptop | env `HARNESS_DIR`, `HARNESS_CALIBRATION`, `DREAM_CHECKPOINT`, `HARNESS_TIME_LIMIT` (default 3600), `AGENT_NAME` (default `pah2026-robot-<hostname>`), `MATCH_MODE` (default on) | Launcher and agent settings. |
| laptop | env `VISION_MODEL` (default `google/gemini-3.6-flash`), `VISION_FALLBACK_MODEL` (default `google/gemini-3.8-flash`; `off` disables it) | Camera answers: the first model gets 6 s; if it fails, times out or answers empty, the second gets the rest of the 12 s. |
| browser | LiveKit Agent Console, agent `pah2026-robot-<hostname>` | Talk to the agent. |
| nuck-3 | `tmux attach -t match`, then **Ctrl-C** | Ends the run: home, park, off. Ctrl-C a second time during start-up forces an exit. |
| nuck-3 | `kill -INT <pid>` (pid: `readlink ~/.dreamlabs/current-harness-run` → `state.json` `pid`) | The same as Ctrl-C. |
| nuck-3 | `kill -USR1 <pid>` | Freeze, the same as the voice word. |

**Not available:** a voice "go home and hold", or resume after a freeze, in harness mode. `dreamcli robot down` won't
work while the harness holds the arms lock.
