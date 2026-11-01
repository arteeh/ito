#!/bin/sh
# Start the virtual Microduck with its MuJoCo viewer: tools/sim.sh
# Connect Ito to localhost:8081 with the printed pairing code. Closing the viewer (or Ctrl+C) stops it.
set -eu
cd "$(dirname "$0")/.."
mkdir -p ~/.config/ito ~/.local/state/ito/microduck
git submodule update --init --depth 1 -q
ITO_SIM_IMAGE=$( { git submodule status third_party; cat drivers/microduck/Dockerfile \
    drivers/microduck/pyproject.toml drivers/microduck/uv.lock; } | sha256sum | cut -c1-12)
export ITO_SIM_IMAGE
docker compose up -d --force-recreate --quiet-pull
docker images ito-microduck --format '{{.Tag}}' | grep -vx -e "$ITO_SIM_IMAGE" -e latest \
    | sed 's/^/ito-microduck:/' | xargs -r docker rmi -f >/dev/null
uv run -q --extra sim python -m drivers.microduck.sim --viewer &
sim=$!
trap 'kill $sim 2>/dev/null || true; docker compose down' EXIT
trap 'exit 130' INT TERM
until docker logs ito-microduck-1 2>&1 | grep -q "Virtual Microduck ready"; do
    kill -0 $sim 2>/dev/null || { echo "the sim body stopped before the robot was ready"; exit 1; }
    sleep 0.5
done
echo "Virtual Microduck ready: connect Ito to localhost:8081, pairing code $(cat ~/.config/ito/microduck-sim-code)"
wait $sim || true
