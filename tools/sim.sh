#!/bin/sh
# Start the virtual Microduck with its MuJoCo viewer, and a Windows build of Ito piloting it:
#   tools/sim.sh [BUILD]          (default: this checkout's branch, as tools/build.sh names it)
# BUILD is a folder in %USERPROFILE%\Projects\ito\builds; pass - to start no Ito. Close Ito and
# start it again as often as you like; closing the viewer (or Ctrl+C) stops the sim.
set -eu
cd "$(dirname "$0")/.."
build=${1:-$(git rev-parse --abbrev-ref HEAD | sed 's#/#-#g')}
mkdir -p ~/.config/ito ~/.local/state/ito/microduck
git submodule update --init --depth 1 -q
docker compose up -d --build --force-recreate --quiet-pull
uv run -q --extra sim python -m drivers.microduck.sim --viewer &
sim=$!
trap 'kill $sim 2>/dev/null || true; docker compose down' EXIT
trap 'exit 130' INT TERM
until docker logs ito-microduck-1 2>&1 | grep -q "Virtual Microduck ready"; do
    kill -0 $sim 2>/dev/null || { echo "the sim body stopped before the robot was ready"; exit 1; }
    sleep 0.5
done
address=$(hostname -I | cut -d' ' -f1):8081
code=$(cat ~/.config/ito/microduck-sim-code)
echo "Virtual Microduck at $address, pairing code $code"
if [ "$build" != - ]; then
    win=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")/Projects/ito
    (cd "$win/builds/$build/ito" && cmd.exe /c start "" ito.exe "$address" --code "$code")
fi
wait $sim || true
