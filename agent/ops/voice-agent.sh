#!/usr/bin/env bash
# Launch the pah2026 voice agent against the real Dream station (through the local SSH tunnels).
# HUMAN-ONLY for --live. Agents must never run `--live`.
set -euo pipefail

PLACEHOLDER="dryrun-placeholder-not-a-model"
STATION_TUNNEL_URL="http://127.0.0.1:8091"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<USAGE
Usage: ops/voice-agent.sh [--backend station|harness|fake] [--check | --dry-run | --live] [--checkpoint ID] [--help]

  --check            preflight only (tunnels, station /status, checkpoint, cameras), then exit
  --dry-run          (default) real station for status and stop; start/home stay dry-run (a task change stops the run, then asks before starting)
  --live             station: real motion (STATION_LIVE=1). Human only. Needs a real checkpoint, station state "idle",
                     a TTY, and typing exactly: E-STOP IN HAND
  --checkpoint ID    Dream checkpoint id (else env DREAM_CHECKPOINT; dry-run falls back to a placeholder)
  --backend B        station (default): the Dream station HTTP API through the tunnels.
                     harness: drive the teammate's \`harness run\` on HARNESS_HOST over SSH (ROBOT_BACKEND=harness).
                       --check: ssh reachable, helper status (lock free, no holders), \`harness run --help\` contract.
                       --live: also needs a real checkpoint and all of --check passing; no station /status needed.
                       Env: HARNESS_HOST (nuck-3), HARNESS_DIR (~/team-harness-voice),
                       HARNESS_CALIBRATION (~/team-harness/harness/calibration.yaml).
                     fake: tests/fake_harness.py run locally (no SSH, no robot) under PAH_FAKE_HOME
                       (default \${XDG_RUNTIME_DIR:-\$HOME/.cache}/pah2026-fake-harness). --live is refused.

NOTE: --backend harness (any mode, including --check) contacts HARNESS_HOST over SSH.

Harness/fake env: HARNESS_TIME_LIMIT (seconds, default 3600; the run does NOT end by itself), AGENT_NAME (default pah2026-robot-\$(hostname)).
Env: STATION_URL (default $STATION_TUNNEL_URL; --live requires exactly that), DREAM_CHECKPOINT.
Always sets: CAMERA_SOURCE=auto CAMERA_SOURCE_URL=http://127.0.0.1:8080 CAMERA_STATION_URL=http://127.0.0.1:8093
USAGE
}

mode="dry-run"
backend="station"
checkpoint="${DREAM_CHECKPOINT:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) mode="check" ;;
    --dry-run) mode="dry-run" ;;
    --live) mode="live" ;;
    --backend)
      [ $# -ge 2 ] || { echo "--backend needs a value" >&2; exit 2; }
      case "$2" in station|harness|fake) backend="$2" ;; *) echo "unknown backend: $2 (station, harness or fake)" >&2; exit 2 ;; esac
      shift ;;
    --checkpoint)
      [ $# -ge 2 ] || { echo "--checkpoint needs a value" >&2; exit 2; }
      checkpoint="$2"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

STATION_URL="${STATION_URL:-$STATION_TUNNEL_URL}"
CAMERA_SOURCE_URL="http://127.0.0.1:8080"
CAMERA_STATION_URL="http://127.0.0.1:8093"

listening() { ss -ltn 2>/dev/null | awk -v p=":$1" '$4 ~ p"$" {f=1} END {exit !f}'; }

tunnel_hint() {  # port, label
  if listening "$1"; then
    echo "  [ok]   127.0.0.1:$1 listening ($2)"
  else
    echo "  [DOWN] 127.0.0.1:$1 not listening ($2). Open the tunnels: ops/tunnels.sh up"
  fi
}

STATUS_STATE=""
preflight() {
  echo "== tunnels"
  tunnel_hint 8091 "station API"
  tunnel_hint 8093 "station view"
  tunnel_hint 8080 "kit dashboard"

  echo "== station status (GET /status, read-only)"
  local body
  if body="$(curl -sf -m 3 "$STATION_URL/status" 2>/dev/null)"; then
    printf '%s' "$body" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print("  state=%s checkpoint=%s arms=%s last_error=%s" % (d.get("state"), d.get("checkpoint"), d.get("arms"), d.get("last_error")))' \
      || echo "  (could not parse /status reply)"
    STATUS_STATE="$(printf '%s' "$body" | python3 -c 'import json,sys
print(json.load(sys.stdin).get("state") or "")' 2>/dev/null || true)"
  else
    echo "  station down: no answer from $STATION_URL/status (is 'dreamcli robot up' running and the tunnel open?)"
    STATUS_STATE=""
  fi

  echo "== checkpoint"
  if [ -z "$checkpoint" ]; then
    echo "  DREAM_CHECKPOINT / --checkpoint NOT set"
  elif [ "$checkpoint" = "$PLACEHOLDER" ]; then
    echo "  placeholder checkpoint (not a real model)"
  else
    echo "  set: $checkpoint"
  fi

  echo "== cameras"
  if curl -sf -m 3 -o /dev/null "http://127.0.0.1:8080/api/camera/overhead.jpg"; then
    echo "  [ok]   kit dashboard overhead.jpg answers on :8080 (dhk-collect running)"
  else
    echo "  [down] kit dashboard :8080 (needs dhk-collect running)"
  fi
  if curl -sf -m 3 -o /dev/null "$CAMERA_STATION_URL/trips"; then
    echo "  [ok]   station view /trips answers on :8093 (Dream run/view active)"
  else
    echo "  [down] station view :8093 (needs the Dream station agent running)"
  fi
}

# ---------------------------------------------------------------- harness / fake backends
HARNESS_PY_REMOTE="~/dream-env/bin/python"   # matches HarnessClient's default python
HARNESS_HELPER="$REPO/src/harness_remote.py"
if [ "$backend" = "fake" ]; then
  FAKE_HOME="${PAH_FAKE_HOME:-${XDG_RUNTIME_DIR:-$HOME/.cache}/pah2026-fake-harness}"
  case "$FAKE_HOME" in
    "$HOME"|"$HOME"/|/|"") echo "refusing PAH_FAKE_HOME='$FAKE_HOME'" >&2; exit 2 ;;
    "$REPO"|"$REPO"/*) echo "refusing PAH_FAKE_HOME inside the repo: $FAKE_HOME" >&2; exit 2 ;;
  esac
  HARNESS_HOST="local"
  HARNESS_DIR="$FAKE_HOME"
  HARNESS_CALIBRATION="$FAKE_HOME/cal.yaml"
  FAKE_CMD="$(command -v python3) $REPO/tests/fake_harness.py"
elif [ "$backend" = "harness" ]; then
  HARNESS_HOST="${HARNESS_HOST:-nuck-3}"
  HARNESS_DIR="${HARNESS_DIR:-~/team-harness-voice}"
  HARNESS_CALIBRATION="${HARNESS_CALIBRATION:-~/team-harness/harness/calibration.yaml}"
  [[ "$HARNESS_DIR" =~ ^[A-Za-z0-9_./~-]+$ ]] || { echo "invalid HARNESS_DIR: $HARNESS_DIR" >&2; exit 2; }
  [[ "$HARNESS_CALIBRATION" =~ ^[A-Za-z0-9_./~-]+$ ]] || { echo "invalid HARNESS_CALIBRATION: $HARNESS_CALIBRATION" >&2; exit 2; }
  [[ "$HARNESS_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]] || { echo "invalid HARNESS_HOST: $HARNESS_HOST" >&2; exit 2; }
fi
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=3 -o ControlPath=none -o ServerAliveInterval=2 -o ServerAliveCountMax=2)
# The match clock: the run does NOT end by itself unless this is shortened. Default 1 hour.
HARNESS_TIME_LIMIT="${HARNESS_TIME_LIMIT:-3600}"
{ [[ "$HARNESS_TIME_LIMIT" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [ "${HARNESS_TIME_LIMIT%%.*}" -gt 0 ]; } || { echo "invalid HARNESS_TIME_LIMIT: $HARNESS_TIME_LIMIT" >&2; exit 2; }
# Extra `harness run` flags (mirror the teammates' command line), e.g. "--color orange --lower-unplug-cm 8.5".
# Shell-safe charset here; the helper validates every token against its allowlist (--check, and at every start).
HARNESS_RUN_ARGS="${HARNESS_RUN_ARGS:-}"
[[ "$HARNESS_RUN_ARGS" =~ ^[A-Za-z0-9\ ._:/,=-]*$ ]] || { echo "invalid HARNESS_RUN_ARGS: $HARNESS_RUN_ARGS" >&2; exit 2; }
# The LiveKit agent name: unique per laptop so two agents on one project never share dispatch.
AGENT_NAME="${AGENT_NAME:-pah2026-robot-$(hostname)}"
[[ "$AGENT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid AGENT_NAME: $AGENT_NAME" >&2; exit 2; }

fake_setup() {  # a stub `python` whose `-m harness run --help` prints the contract; ~ expands to FAKE_HOME
  mkdir -p "$FAKE_HOME/dream-env/bin"
  rm -f "$FAKE_HOME/dream-env/bin/python"
  cat >"$FAKE_HOME/dream-env/bin/python" <<'STUB'
#!/bin/sh
printf 'usage: harness run\nSIGUSR1 freezes the arms\nomit to bin every ball except X\n'
STUB
  chmod +x "$FAKE_HOME/dream-env/bin/python"
  touch "$FAKE_HOME/cal.yaml"
}

helper_check() {  # one JSON line; ssh for harness, local python for fake
  if [ "$backend" = "fake" ]; then
    HARNESS_REMOTE_HOME="$FAKE_HOME" HARNESS_FAKE_CMD="$FAKE_CMD" \
      python3 - check --dir "$HARNESS_DIR" --python "$HARNESS_PY_REMOTE" --calibration "$HARNESS_CALIBRATION" --extra-args="$HARNESS_RUN_ARGS" <"$HARNESS_HELPER"
  else
    timeout 30 ssh "${SSH_OPTS[@]}" -- "$HARNESS_HOST" \
      "python3 - check --dir '$HARNESS_DIR' --python '$HARNESS_PY_REMOTE' --calibration '$HARNESS_CALIBRATION' --extra-args='$HARNESS_RUN_ARGS'" <"$HARNESS_HELPER"
  fi
}

HARNESS_CHECK_OK=0      # 1 only when every check passed
HARNESS_FREE=0          # 1 when lock free, no holders, no run
harness_preflight() {
  HARNESS_CHECK_OK=0; HARNESS_FREE=0
  echo "== backend: $backend (host $HARNESS_HOST, dir $HARNESS_DIR)"
  echo "== ssh"
  if [ "$backend" = "fake" ]; then
    echo "  [skip] fake backend runs locally (HARNESS_REMOTE_HOME=$FAKE_HOME)"
    fake_setup
  elif timeout 30 ssh "${SSH_OPTS[@]}" -- "$HARNESS_HOST" true >/dev/null 2>&1; then
    echo "  [ok]   ssh $HARNESS_HOST reachable"
  else
    echo "  [FAIL] ssh $HARNESS_HOST not reachable (BatchMode, 3 s)"
    return 0
  fi
  if [ "$backend" = "harness" ]; then tunnel_hint 8093 "station view (DebugView)"; fi
  echo "== remote helper (status + harness run --help contract)"
  local out
  if ! out="$(helper_check 2>/dev/null)" || [ -z "$out" ]; then
    echo "  [FAIL] helper did not answer"
    return 0
  fi
  local lines
  lines="$(printf '%s' "$out" | python3 -c 'import json,sys
d=json.load(sys.stdin)
if not d.get("ok"):
    print("FAIL helper error: %s" % d.get("error")); sys.exit(0)
print("%s contract: %s" % ("ok" if d["contract_ok"] else "FAIL", "harness run --help has the freeze contract" if d["contract_ok"] else (d.get("help_error") or "help missing: " + "; ".join(d["missing"]))))
xa = d.get("extra_args_ok", True)
print("%s run args: %s" % ("ok" if xa else "FAIL", "HARNESS_RUN_ARGS accepted" if xa else d.get("extra_args_error")))
free = d["lock_free"] and not d["holders"] and not d["untracked_pids"] and not d["running"] and d.get("match_session") != "live"
print("%s arms: lock_free=%s holders=%s untracked_pids=%s run_active=%s match_session=%s" % ("ok" if free else "FAIL", d["lock_free"], d["holders"], d["untracked_pids"], d["running"], d.get("match_session")))
cal = d.get("calibration_ok")
print("%s calibration: %s" % ("ok" if cal is not False else "FAIL", "present" if cal else ("MISSING on host" if cal is False else "not checked")))
print("FREE" if free and cal is not False else "BUSY")
print("CONTRACT" if d["contract_ok"] and xa else "NOCONTRACT")' 2>/dev/null)" || { echo "  [FAIL] could not parse helper reply"; return 0; }
  printf '%s\n' "$lines" | grep -v -e '^FREE$' -e '^BUSY$' -e '^CONTRACT$' -e '^NOCONTRACT$' | sed -e 's/^ok /  [ok]   /' -e 's/^FAIL /  [FAIL] /'
  if printf '%s\n' "$lines" | grep -qx FREE; then HARNESS_FREE=1; fi
  if printf '%s\n' "$lines" | grep -qx FREE && printf '%s\n' "$lines" | grep -qx CONTRACT; then HARNESS_CHECK_OK=1; fi
  echo "== checkpoint"
  if [ -z "$checkpoint" ]; then echo "  DREAM_CHECKPOINT / --checkpoint NOT set"
  elif [ "$checkpoint" = "$PLACEHOLDER" ]; then echo "  placeholder checkpoint (not a real model)"
  else echo "  set: $checkpoint"; fi
}

if [ "$backend" != "station" ]; then
  if [ "$mode" = "check" ]; then
    harness_preflight
    [ "$HARNESS_CHECK_OK" = 1 ] || { echo "CHECK FAILED" >&2; exit 1; }
    echo "CHECK OK"
    exit 0
  fi
  if [ "$mode" = "dry-run" ]; then
    if [ -z "$checkpoint" ]; then
      checkpoint="$PLACEHOLDER"
      if [ "$backend" = "harness" ]; then
        checkpoint=""   # never a placeholder: a --checkpoint would force one (fake) model onto every stage
        echo "NOTE: no checkpoint given; 'harness run --dry-run' uses the harness's per-stage default models (calls Dream inference; arms read-only)."
      else
        echo "NOTE: no checkpoint given; using placeholder '$PLACEHOLDER'. Dry-run only; it cannot start a real run."
      fi
    fi
    harness_preflight || true
    if [ "$backend" = "fake" ]; then
      echo "== FAKE: start/end/freeze run against the local fake harness (nothing moves)."
    else
      echo "== DRY-RUN: start the relay launches harness run --dry-run on the station: it opens the arms read-only and takes every camera. Arms resting or supported; nothing else may use the arms or cameras."
    fi
  fi
  if [ "$mode" = "live" ]; then
    fail() { echo "REFUSING --live: $*" >&2; exit 1; }
    [ "$backend" = "harness" ] || fail "backend '$backend' has nothing live"
    harness_preflight || true
    [ "$checkpoint" != "$PLACEHOLDER" ] || fail "the placeholder checkpoint cannot go live"
    [ "$HARNESS_CHECK_OK" = 1 ] || fail "the harness checks did not all pass (see above)"
    [ "$HARNESS_FREE" = 1 ] || fail "the arms have a holder (station, dhk-collect or another run); stop it first"
    [ -t 0 ] || fail "stdin is not a TTY"
    [ -r /dev/tty ] && [ -w /dev/tty ] || fail "no controlling terminal (/dev/tty)"
    cat >/dev/tty <<'BANNER'

  ============================ LIVE MOTION ============================
  * Emergency words: FREEZE or HALT (no confirmation). No physical e-stop: the emergency word holds the arms; Ctrl-C in the station's harness window ends the run.
  * Starting the relay moves the arms as soon as you say yes.
  * After "go" in a match the voice agent must stay silent.
  * The run does NOT end by itself; freeze, let the referee score, then end the run.
  =====================================================================
BANNER
    printf 'Type exactly "E-STOP IN HAND" to enable live motion: ' >/dev/tty
    IFS= read -r answer </dev/tty || fail "could not read confirmation"
    [ "$answer" = "E-STOP IN HAND" ] || fail "confirmation did not match"
    echo "== LIVE motion enabled (HARNESS_LIVE=1)"
  fi
  cd "$REPO"
  # harness dry-run without a real model: never forward the placeholder to `harness run` (it has its own default)
  agent_checkpoint="$checkpoint"
  if [ "$backend" = "harness" ] && [ "$checkpoint" = "$PLACEHOLDER" ]; then agent_checkpoint=""; fi
  extra_env=()
  if [ "$backend" = "fake" ]; then extra_env=(HARNESS_REMOTE_HOME="$FAKE_HOME" HARNESS_FAKE_CMD="$FAKE_CMD"); fi
  if [ "$mode" = "live" ] && [ "$backend" = "harness" ]; then extra_env+=(HARNESS_LIVE=1); fi
  echo "Emergency words: FREEZE or HALT (no confirmation). No physical e-stop: the emergency word holds the arms; Ctrl-C in the station's harness window ends the run."
  echo "Agent name: $AGENT_NAME"
  if [ "$backend" = "harness" ]; then
    echo "Run args: HARNESS_RUN_ARGS='${HARNESS_RUN_ARGS}'"
    if [ -n "$agent_checkpoint" ]; then echo "checkpoint: $agent_checkpoint (one model for all stages)"; else echo "checkpoint: per-stage defaults"; fi
  fi
  echo "Match clock: HARNESS_TIME_LIMIT=${HARNESS_TIME_LIMIT} s: the run does NOT end by itself; freeze, let the referee score, then end the run."
  (
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.config/pah2026/secrets.sh"
    set +a
    exec env -u STATION_LIVE -u HARNESS_REMOTE_HOME -u HARNESS_FAKE_CMD \
      HARNESS_LIVE=0 \
      ROBOT_BACKEND="$backend" \
      HARNESS_HOST="$HARNESS_HOST" \
      HARNESS_DIR="$HARNESS_DIR" \
      HARNESS_CALIBRATION="$HARNESS_CALIBRATION" \
      HARNESS_TIME_LIMIT="$HARNESS_TIME_LIMIT" \
      HARNESS_RUN_ARGS="$HARNESS_RUN_ARGS" \
      AGENT_NAME="$AGENT_NAME" \
      DREAM_CHECKPOINT="$agent_checkpoint" \
      CAMERA_SOURCE=auto \
      CAMERA_SOURCE_URL="$CAMERA_SOURCE_URL" \
      CAMERA_STATION_URL="$CAMERA_STATION_URL" \
      "${extra_env[@]}" \
      lk agent dev --no-reload
  )
  exit 0
fi

if [ "$mode" = "check" ]; then
  preflight
  exit 0
fi

if [ "$mode" = "dry-run" ]; then
  if [ -z "$checkpoint" ]; then
    checkpoint="$PLACEHOLDER"
    echo "NOTE: no checkpoint given; using placeholder '$PLACEHOLDER'. Dry-run only; it cannot start a real run."
  fi
  preflight || true
  if [ -z "$STATUS_STATE" ]; then echo "WARNING: station is down; continuing (status/stop will report errors)."; fi
  echo "== starting in DRY-RUN: start/home are NOT sent; status and stop are real."
fi

if [ "$mode" = "live" ]; then
  preflight || true
  fail() { echo "REFUSING --live: $*" >&2; exit 1; }
  { [ -n "$checkpoint" ] && [ "$checkpoint" != "$PLACEHOLDER" ]; } || fail "a real checkpoint is required (--checkpoint ID or DREAM_CHECKPOINT)"
  [ "$STATION_URL" = "$STATION_TUNNEL_URL" ] || fail "STATION_URL must be $STATION_TUNNEL_URL (the localhost tunnel), got '$STATION_URL'"
  [ -n "$STATUS_STATE" ] || fail "GET $STATION_URL/status failed (station down or tunnel closed)"
  # station.py defines IDLE="idle" (also starting/running/stopping/homing/resetting/error); there is no "ready".
  [ "$STATUS_STATE" = "idle" ] || fail "station state is '$STATUS_STATE', must be 'idle'"
  [ -t 0 ] || fail "stdin is not a TTY"
  [ -r /dev/tty ] && [ -w /dev/tty ] || fail "no controlling terminal (/dev/tty)"
  cat >/dev/tty <<'BANNER'

  ============================ LIVE MOTION ============================
  * No voice command works during a home move; it cannot be interrupted.
  * Emergency words: FREEZE or HALT (no confirmation). No physical e-stop: the emergency word holds the arms; Ctrl-C in the station's harness window ends the run.
  * "stop" goes through the agent and is slower.
  * "Yes" means NOW: the arm moves as soon as you confirm.
  * Make the first task short and harmless.
  =====================================================================
BANNER
  printf 'Type exactly "E-STOP IN HAND" to enable live motion: ' >/dev/tty
  IFS= read -r answer </dev/tty || fail "could not read confirmation"
  [ "$answer" = "E-STOP IN HAND" ] || fail "confirmation did not match"
  echo "== LIVE motion enabled (STATION_LIVE=1)"
fi

cd "$REPO"
live_flag=()
if [ "$mode" = "live" ]; then live_flag=(STATION_LIVE=1); fi
# Secrets are sourced only inside this subshell; nothing is echoed. env -u clears any inherited STATION_LIVE.
(
  set -a
  # shellcheck disable=SC1091
  . "$HOME/.config/pah2026/secrets.sh"
  set +a
  exec env -u STATION_LIVE \
    HARNESS_LIVE=0 \
    STATION_URL="$STATION_URL" \
    DREAM_CHECKPOINT="$checkpoint" \
    ROBOT_BACKEND=station \
    CAMERA_SOURCE=auto \
    CAMERA_SOURCE_URL="$CAMERA_SOURCE_URL" \
    CAMERA_STATION_URL="$CAMERA_STATION_URL" \
    "${live_flag[@]}" \
    lk agent dev --no-reload
)
