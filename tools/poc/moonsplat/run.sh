#!/bin/bash
# run.sh TAPE CELL [START END]: MoonSplat (run.py) on a sim tape in the ito-research/moonsplat
# image, under the Ceres session lock, the container capped at 6 GB RAM. nvidia-smi is sampled
# into vram.csv meanwhile (the whole card: other jobs cannot run while the lock is held).
# Output ~/Projects/ito/out/moonsplat/runs/TAPE__CELL. Env: REFINE=1 for upstream's final colour
# refinement; TOOLS: tools/tapes checkout (default ~/Projects/ito/work/tapes/tools/tapes).
set -u
tape=$1; cell=$2; start=${3:-0}; end=${4:-60}
HERE=$(cd "$(dirname "$0")" && pwd)
R=$HOME/Projects/ito/research
TOOLS=${TOOLS:-$HOME/Projects/ito/work/tapes/tools/tapes}
LOCK=$HOME/.unii/work/ceres-session.lock
out=$HOME/Projects/ito/out/moonsplat/runs/${tape}__${cell}
rm -rf "$out"; mkdir -p "$out"
export tape cell start end HERE R TOOLS out
flock "$LOCK" bash -c '
    [ -e "$HOME/.unii/work/USER-ON-CERES" ] && { echo "user on Ceres: not starting"; exit 75; }
    ( while true; do echo "$(date +%s.%N),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 2; done ) > "$out/vram.csv" &
    sampler=$!
    ( sleep 3; while docker inspect "moonsplat-$cell" >/dev/null 2>&1; do echo "$(date +%s),$(docker stats --no-stream --format "{{.MemUsage}}" "moonsplat-$cell" 2>/dev/null)"; sleep 1; done ) > "$out/ram.csv" &
    echo "$(date +%T) moonsplat $tape $cell $start-$end"
    docker run --rm --name "moonsplat-$cell" --gpus all --ipc=host --memory ${MEM:-6g} --memory-swap ${MEM:-6g} \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$R/cache:/tmp/.cache" \
        -v "$R/systems/mast3r-slam/checkpoints:/data/pug/MASt3R-SLAM/checkpoints:ro" \
        -v "$R/data/$tape:/tape:ro" -v "$TOOLS:/tools:ro" -v "$HERE:/poc:ro" -v "$out:/out" \
        -w /opt/moonsplat ito-research/moonsplat:poc \
        python /poc/run.py /tape /out --tools /tools --cell "$cell" --start "$start" --end "$end" \
        ${REFINE:+--refine} > "$out/run.log" 2>&1
    rc=$?; kill $sampler; exit $rc'
rc=$?
echo "$(date +%T) rc=$rc"; grep -v -E "it/s|Warning|warn" "$out/run.log" | tail -n 4
exit $rc
