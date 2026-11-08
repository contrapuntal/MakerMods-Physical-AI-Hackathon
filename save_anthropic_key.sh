#!/usr/bin/env bash
# Asks for the Anthropic key once (hidden) and saves it on the station for the harness's vision checks:
# ~/team-harness/.anthropic_key, readable only by the pah account. Nothing is printed.
set -euo pipefail
read -rsp "Paste the Anthropic key, then press Return: " k; echo
[[ "$k" == sk-ant-* ]] || { echo "That doesn't look like an Anthropic key (sk-ant-...); nothing saved."; exit 1; }
printf '%s\n' "$k" | ssh pah@10.103.0.125 'umask 077; mkdir -p ~/team-harness; IFS= read -r k; printf "%s\n" "$k" > ~/team-harness/.anthropic_key; chmod 600 ~/team-harness/.anthropic_key; echo "station: saved"'
unset k
