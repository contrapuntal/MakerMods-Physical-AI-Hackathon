# Runs ON THE STATION (send it with: ssh pah@10.103.0.125 bash -s < station_install_key.sh).
# Moves the Dream API key from ~/apikey.txt into ~/.dreamlabs/credentials without printing it, keeps the platform
# URLs, backs up the old file, then deletes the plain-text ~/apikey.txt.
set -eu
src=~/apikey.txt
f=~/.dreamlabs/credentials
[ -f "$src" ] || { echo "no ~/apikey.txt on $(hostname)"; exit 1; }
k=$(grep -oE 'dl_live_[A-Za-z0-9_-]+' "$src" | head -1)
[ -n "$k" ] || { echo "no dl_live_ key in ~/apikey.txt"; exit 1; }
cp "$f" "$f.bak"
{ grep -v '^DREAM_API_KEY=' "$f.bak"; printf 'DREAM_API_KEY=%s\n' "$k"; } > "$f"
chmod 600 "$f"
unset k
rm -f "$src"
echo "saved on $(hostname); ~/apikey.txt removed"
