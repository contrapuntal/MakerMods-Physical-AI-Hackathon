#!/usr/bin/env bash
# One-shot environment install for the MakerMods Physical AI Hackathon (Nov 7–8, 2026).
# Every repo pins Python 3.12 (dreamlabs needs <3.14, portal-makermods needs 3.12.x).
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)/repos"

echo "== cloning upstream repos (skips any already present)"
mkdir -p "$R"
for url in \
  https://github.com/livekit-examples/portal-makermods.git \
  https://github.com/makermods-robotics/maker-arm-sdk.git \
  https://github.com/makermods-robotics/makermodslab.git \
  https://github.com/makermods-robotics/metal-arm-harness.git \
  https://github.com/makermods-robotics/metal-python-ros.git; do
  d="$R/$(basename "$url" .git)"
  [[ -d "$d/.git" ]] || git clone "$url" "$d"
done

echo "== portal-makermods (LiveKit Portal + Dream-1 + LeRobot + ACT)"
(cd "$R/portal-makermods" && uv sync --python 3.12)

echo "== maker-arm-sdk (Maker Arm v1 CAN driver + CLI)"
(cd "$R/maker-arm-sdk" && uv venv -q --python 3.12 && uv pip install -e ".[dev]")

echo "== makermodslab (web UI: calibrate / teleop / record / train)"
(cd "$R/makermodslab" && uv venv -q --python 3.12 && uv pip install -e .)

echo "== metal-arm-harness (safe agent-driven arm control, sim backend available)"
(cd "$R/metal-arm-harness" && uv venv -q --python 3.12 && uv pip install -e .)

echo "== quick checks"
(cd "$R/portal-makermods" && uv run python -c "import dream, livekit, lerobot; print('portal env OK')")
"$R/maker-arm-sdk/.venv/bin/maker-arm" doctor || true
lk --version
echo "All done. Next: fill in repos/portal-makermods/.env (LiveKit + DREAM_API_KEY)."
