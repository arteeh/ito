#!/bin/bash
# tapevid.sh TAPE CANDIDATE CODE [ITO_X=1 ...]: film CODE's map on TAPE (see replay.py), then
# compose the video (compose.py). Run in WSL on Ceres. Takes ~/.unii/work/ceres-session.lock
# for the replay only; refuses from 19:55 and while USER-ON-CERES exists.
#   TAPE       name under ~/Projects/ito/research/data (or an absolute WSL path)
#   CANDIDATE  label for file names: <tape>__<candidate>
#   CODE       a git ref of ~/Projects/ito/main (snapshotted with git archive into
#              ~/Projects/ito/out/tapes/code/<sha>) or an absolute path to a checkout
# Options (environment): BUILD=feat-mast3r-slam-full (whose Python runs it; needs gsplat),
#   TRUTH=poses (true-pose reference), END=seconds, FPS=15, DROP=name (also copy the mp4 to
#   C:\Users\me\Projects\ito\videos\<drop>\), REF=<candidate label of the reference run> (default ref).
set -u
tape=$1; cand=$2; code=$3; shift 3
envs="$*"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$HOME/Projects/ito/main
DATA=$HOME/Projects/ito/research/data
LOCK=$HOME/.unii/work/ceres-session.lock
BUILD=${BUILD:-feat-mast3r-slam-full}
FPS=${FPS:-15}
case "$tape" in /*) tapedir=$tape; tape=$(basename "$tape") ;; *) tapedir=$DATA/$tape ;; esac
[ -f "$tapedir/frames.jsonl" ] || { echo "no tape at $tapedir"; exit 2; }
name="${tape}__${cand}"
RUNS=/mnt/c/Users/me/Projects/ito/out/tapes/runs
out=$RUNS/$name
mkdir -p "$out"

if [ -d "$code" ]; then
    root=$code
else
    sha=$(git -C "$REPO" rev-parse --verify -q "$code^{commit}") || { echo "unknown ref $code"; exit 2; }
    root=$HOME/Projects/ito/out/tapes/code/$sha
    if [ ! -d "$root" ]; then
        mkdir -p "$root.tmp" && git -C "$REPO" archive "$sha" | tar -x -C "$root.tmp" && mv "$root.tmp" "$root"
    fi
    echo "$code = $sha" > "$out/code.txt"
fi
echo "$envs" > "$out/env.txt"

late() { [ "$(date +%H%M)" -ge 1955 ] || [ "$(date +%H%M)" -lt 0700 ]; }
win() { wslpath -w "$1"; }
pre=""; for pair in $envs; do pre="$pre set $pair&&"; done
args="$(win "$tapedir") $(win "$out") --code $(win "$root") --fps $FPS"
[ -n "${TRUTH:-}" ] && args="$args --truth $TRUTH"
[ -n "${END:-}" ] && args="$args --end $END"
script=$(win "$HERE/replay.py")
export pre args script BUILD out
while true; do
    late && { echo "$(date +%T) not starting: Ceres is the user's from 19:55"; exit 75; }
    flock "$LOCK" bash -c '
        [ -e "$HOME/.unii/work/USER-ON-CERES" ] && exit 75
        ss -ltn | grep -q ":8081 " && exit 75
        n=$(cd /mnt/c && powershell.exe -NoProfile -Command "(Get-Process -ErrorAction SilentlyContinue | ? {\$_.ProcessName -match \"^(ito)$\"}).Count" 2>/dev/null < /dev/null | tr -d "\r ")
        [ "${n:-0}" != "0" ] && exit 75
        echo "$(date +%T) replay $out"
        cd /mnt/c/Users/me/Projects/ito/builds/$BUILD/ito && cmd.exe /c "$pre runtime\\python.exe -I $script $args" > "$out/replay.log" 2>&1 < /dev/null
        rc=$?; [ $rc = 75 ] && rc=76; exit $rc'
    rc=$?
    [ $rc = 75 ] && { echo "$(date +%T) Ceres busy, waiting"; sleep 30; continue; }
    break
done
echo "$(date +%T) replay rc=$rc"; tail -2 "$out/replay.log"
[ $rc = 0 ] || exit $rc
REF=${REF:-ref}
"$HOME/.local/bin/uv" run -q --no-project --with opencv-python-headless --with imageio-ffmpeg --with numpy \
    python "$HERE/compose.py" "$tapedir" "$out" --ref "$RUNS/${tape}__$REF" ${DROP:+--drop "$DROP"}
