#!/usr/bin/env bash
# Open, check or close the laptop's SSH tunnels to the Dream station on nuck-3.
#
# One SSH control connection carries every forward, so "down" closes them all together and never touches other
# ssh processes. Forwards bind 127.0.0.1 only. A tunnel moves nothing: it only makes nuck-3's localhost ports
# reachable from this laptop. "status" sends one read-only GET per port (GET /status on the station API).
set -euo pipefail

HOST="${TUNNEL_HOST:-nuck-3}"
SOCK="${XDG_RUNTIME_DIR:-/tmp}/pah2026-tunnels-${HOST}.sock"
DEFAULT_PORTS=(8080 8091 8093)

usage() {
  cat <<EOF
Usage: ops/tunnels.sh up|down|status|fg [PORT...]

  up      open the tunnels in the background (default ports: ${DEFAULT_PORTS[*]})
  down    close every tunnel this script opened
  status  per port: is the tunnel open, who holds it, does nuck-3 answer
  fg      open the tunnels, stay in the foreground, close them on Ctrl-C

Ports: 8080 kit dashboard (dhk-collect or check_cameras), 8091 station API,
       8093 station model view (dreamcli robot up --view), 8092 setup wizard.
Env:   TUNNEL_HOST (default nuck-3)
EOF
}

label() {
  case "$1" in
    8080) echo "kit dashboard" ;;
    8091) echo "station API" ;;
    8092) echo "setup wizard" ;;
    8093) echo "station view" ;;
    *) echo "port $1" ;;
  esac
}

needs() {  # what must run on nuck-3 for this port to answer
  case "$1" in
    8080) echo "dhk-collect (or scripts/check_cameras.py) running on $HOST" ;;
    8091) echo "'dreamcli robot up' running on $HOST" ;;
    8092) echo "'dreamcli setup' running on $HOST" ;;
    8093) echo "'dreamcli robot up --view' running on $HOST" ;;
    *) echo "a service on $HOST:$1" ;;
  esac
}

probe_path() { case "$1" in 8080) echo "/api/camera/overhead.jpg" ;; 8091) echo "/status" ;; *) echo "/" ;; esac; }

master_up() { ssh -S "$SOCK" -O check "$HOST" >/dev/null 2>&1; }
master_pid() { ssh -S "$SOCK" -O check "$HOST" 2>&1 | sed -n 's/.*pid=\([0-9]*\).*/\1/p'; }
holder_pid() { ss -ltnpH "sport = :$1" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1; }
is_listening() { [ -n "$(ss -ltnH "sport = :$1" 2>/dev/null)" ]; }
cmdline() { ps -o args= -p "$1" 2>/dev/null | cut -c1-70 || true; }

check_ports() {
  local p
  for p in "$@"; do
    [[ "$p" =~ ^[0-9]+$ ]] && [ "$p" -ge 1 ] && [ "$p" -le 65535 ] || { echo "not a port: $p" >&2; exit 2; }
  done
}

cmd_status() {
  local mpid p hp owner code
  mpid=""
  if master_up; then
    mpid="$(master_pid)"
    echo "control connection to $HOST: up (pid $mpid)"
  else
    echo "control connection to $HOST: down"
  fi
  for p in "$@"; do
    hp="$(holder_pid "$p")"
    if ! is_listening "$p"; then
      echo "  [closed] :$p $(label "$p")"
      continue
    fi
    if [ -n "$mpid" ] && [ "$hp" = "$mpid" ]; then owner="this script"; else owner="pid ${hp:-?}: $(cmdline "${hp:-0}")"; fi
    code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p$(probe_path "$p")" || true)"
    if [ "$code" = "000" ] || [ -z "$code" ]; then
      echo "  [open]   :$p $(label "$p"), nothing answering yet (needs $(needs "$p")) [$owner]"
    else
      echo "  [ok]     :$p $(label "$p"), HTTP $code [$owner]"
    fi
  done
}

cmd_up() {
  local mpid p hp
  if ! master_up; then
    if ! ssh -f -N -M -S "$SOCK" -o ControlPersist=no -o BatchMode=yes -o ConnectTimeout=8 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$HOST"; then
      echo "could not connect to $HOST" >&2
      exit 1
    fi
  fi
  mpid="$(master_pid)"
  for p in "$@"; do
    if is_listening "$p"; then
      hp="$(holder_pid "$p")"
      if [ "$hp" = "$mpid" ]; then
        echo "  [same]   :$p already open"
      else
        echo "  [skip]   :$p is taken by pid ${hp:-?} ($(cmdline "${hp:-0}")); close that first, e.g. kill ${hp:-<pid>}"
      fi
      continue
    fi
    if ssh -S "$SOCK" -O forward -L "127.0.0.1:$p:localhost:$p" "$HOST" >/dev/null 2>&1; then
      echo "  [added]  :$p $(label "$p")"
    else
      echo "  [fail]   :$p could not be forwarded"
    fi
  done
  cmd_status "$@"
}

cmd_down() {
  local p hp
  if master_up; then
    ssh -S "$SOCK" -O exit "$HOST" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do master_up || break; sleep 0.2; done
    echo "closed the tunnels to $HOST"
  else
    echo "no tunnels open by this script"
  fi
  for p in "$@"; do
    if is_listening "$p"; then
      hp="$(holder_pid "$p")"
      echo "  note: :$p is still held by pid ${hp:-?} ($(cmdline "${hp:-0}")), not opened by this script; stop it with: kill ${hp:-<pid>}"
    fi
  done
}

action="${1:-}"
[ $# -gt 0 ] && shift
ports=("$@")
[ ${#ports[@]} -gt 0 ] || ports=("${DEFAULT_PORTS[@]}")
check_ports "${ports[@]}"

case "$action" in
  up) cmd_up "${ports[@]}" ;;
  down) cmd_down "${ports[@]}" ;;
  status) cmd_status "${ports[@]}" ;;
  fg)
    cmd_up "${ports[@]}"
    trap 'echo; cmd_down "${ports[@]}"; exit 0' INT TERM
    echo "tunnels open; Ctrl-C closes them"
    while master_up; do sleep 5; done
    echo "the control connection to $HOST dropped"
    ;;
  -h | --help | help) usage ;;
  *) usage >&2; exit 2 ;;
esac
