#!/usr/bin/env bash
# Only this script's Xvfb receives injected input; the owner's desktop is untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != --inside ]]; then
    exec xvfb-run -a bash "$0" --inside "${1:-../p48-out/rendered/truth-oracle/tour-world.npz}"
fi
export LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 SDL_VIDEODRIVER=x11
unset WAYLAND_DISPLAY
output="$(realpath -m ../p48-out/viewer-e2e)"
mkdir -p "$output"
rm -f "$output"/frame-*.png
: > "$output/frames.jsonl"
../p48-out/venv/bin/python -m ito.proto.viewcache "$2" --size 960 540 \
    --frames 80 --capture-dir "$output" --metrics "$output/frames.jsonl" > "$output/run.log" 2>&1 &
app=$!
trap 'kill "$app" 2>/dev/null || true' EXIT
# Wait for loading, then move in the actual input loop and capture its display.
for attempt in $(seq 1 100); do
    [[ -s "$output/frames.jsonl" ]] && break
    sleep .2
done
window=$(../p48-out/venv/bin/python -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["window"])' "$output/frames.jsonl")
xdotool windowfocus "$window"
xdotool keydown --window "$window" w
sleep .5
xdotool keyup --window "$window" w
xdotool key --window "$window" F12
sleep 1
xdotool key --window "$window" Home
sleep .2
xdotool mousemove --window "$window" 30 100 click 1
sleep .5
xdotool key --window "$window" F12
sleep .5
wait "$app"
trap - EXIT
../p48-out/venv/bin/python - "$output" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
out=Path(sys.argv[1])
rows=[json.loads(r) for r in (out/'frames.jsonl').read_text().splitlines()]
poses=np.array([r['camera'] for r in rows])
assert np.ptp(poses[:, :3, 3], axis=0).max() > .05, 'Keyboard movement did not reach the renderer'
assert np.allclose(poses[0], poses[-1], atol=.001), 'Home did not recenter'
assert any(r['stereo'] for r in rows), 'ImGui stereo control did not respond'
assert len(list(out.glob('frame-*.png'))) >= 2, 'Screenshots missing'
print(f'PASS: {len(rows)} display frames, keyboard translation, recenter, screenshots')
PY
