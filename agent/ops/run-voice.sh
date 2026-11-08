#!/usr/bin/env bash
# One command to run the hill-climb voice agent on this laptop.
#
#   ops/run-voice.sh            dry run (default): voice start runs `harness run --dry-run`; nothing moves
#   ops/run-voice.sh --check    preflight only (read-only on nuck-3), then exit
#   ops/run-voice.sh --live     live motion: HUMAN ONLY, asks for a typed confirmation; someone at the nuck-3 keyboard
#   ops/run-voice.sh --fake     local test against a fake harness: no station at all
#   ops/run-voice.sh --stop     stop the voice agent running on this laptop (Ctrl-C does the same in its terminal)
#
# It brings the SSH tunnels up, refuses to start a second copy of the agent, sets the Agent Console deployment and
# the teammates' harness flags, then hands over to ops/voice-agent.sh (which loads the secrets and runs its checks).
# Override any setting from the environment, e.g. HARNESS_RUN_ARGS="--color orange" ops/run-voice.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

mode="${1:---dry-run}"
case "$mode" in
  --dry-run|--check|--live|--fake|--stop) ;;
  -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "usage: ops/run-voice.sh [--dry-run|--check|--live|--fake|--stop]" >&2; exit 2 ;;
esac

if [ "$mode" = "--stop" ]; then
  # The agent is `lk agent dev` plus its worker processes. Ctrl-C (SIGINT) first so it leaves its LiveKit session
  # cleanly; anything still alive after 10 s gets SIGTERM. The SSH tunnels stay up (ops/tunnels.sh down closes them).
  pids=$(pgrep -f "lk agent de[v]" || true)
  [ -n "$pids" ] || { echo "no voice agent is running on this laptop"; exit 0; }
  kids=$(for p in $pids; do pgrep -P "$p" || true; done)
  echo "stopping the voice agent (pid $(echo $pids))"
  kill -INT $pids 2>/dev/null || true
  for _ in $(seq 20); do pgrep -f "lk agent de[v]" >/dev/null || break; sleep 0.5; done
  left=$( { pgrep -f "lk agent de[v]" || true; for k in $kids; do kill -0 "$k" 2>/dev/null && echo "$k"; done; } | sort -u)
  if [ -n "$left" ]; then kill -TERM $left 2>/dev/null || true; echo "forced: $(echo $left)"; fi
  echo "stopped"
  exit 0
fi

# One agent per name: a second worker with the same name would split the Agent Console's sessions.
if [ "$mode" != "--check" ] && pgrep -f "lk agent de[v]" >/dev/null; then
  echo "A voice agent is already running on this laptop (pid $(pgrep -f 'lk agent de[v]' | tr '\n' ' ')). Stop it first." >&2
  exit 1
fi

export LIVEKIT_AGENT_DEPLOYMENT="${LIVEKIT_AGENT_DEPLOYMENT:-production}"   # so the LiveKit Agent Console reaches it
# The teammates' last full launch line (read from their run on nuck-3, 14:27): check it with them before a live run.
export HARNESS_RUN_ARGS="${HARNESS_RUN_ARGS---color orange --gravity-scale 0 --gravity-unplug 0.3 --stack-pass 3-cubes --all-colors --pull-cables-until 45 --model-balls dream-1-ft:39478655-nuck3-balls-whites}"
# HARNESS_TIME_LIMIT: default 3600 s (the run ends by voice or Ctrl-C); set 300 for a 5-minute match like theirs.

if [ "$mode" = "--fake" ]; then
  exec ops/voice-agent.sh --backend fake
fi

ops/tunnels.sh up >/dev/null || { echo "could not open the SSH tunnels to nuck-3 (is the laptop on the event network?)" >&2; exit 1; }
echo "tunnels to nuck-3: up    agent: ${AGENT_NAME:-pah2026-robot-$(hostname)}    harness flags: ${HARNESS_RUN_ARGS:-<none>}"
case "$mode" in
  --check)   exec ops/voice-agent.sh --backend harness --check ;;
  --dry-run) exec ops/voice-agent.sh --backend harness ;;
  --live)    exec ops/voice-agent.sh --backend harness --live ;;
esac
