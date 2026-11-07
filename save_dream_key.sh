#!/usr/bin/env bash
# Asks for the Dream API key once (hidden) and saves it on the station and in the local portal .env.
set -euo pipefail
read -rsp "Paste the Dream API key, then press Return: " k; echo
[[ "$k" == dl_live_* ]] || { echo "That doesn't look like a dl_live_ key; nothing saved."; exit 1; }

# Station: replace only the DREAM_API_KEY line, keep the platform URLs, back up the old file.
printf '%s\n' "$k" | ssh pah@10.103.0.125 'IFS= read -r k; f=~/.dreamlabs/credentials; cp "$f" "$f.bak"; { grep -v "^DREAM_API_KEY=" "$f.bak"; printf "DREAM_API_KEY=%s\n" "$k"; } > "$f"; chmod 600 "$f"; echo "station: saved"'

# Mac: the portal-makermods .env (optional, for local dev).
env_file="$(dirname "$0")/repos/portal-makermods/.env"
grep -v '^DREAM_API_KEY=' "$env_file" > "$env_file.tmp" || true
printf 'DREAM_API_KEY=%s\n' "$k" >> "$env_file.tmp"; mv "$env_file.tmp" "$env_file"; chmod 600 "$env_file"
unset k; echo "mac: saved"
