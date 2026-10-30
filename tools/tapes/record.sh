#!/bin/bash
# record.sh NAME SCRIPT.json: record a sim tape ~/Projects/ito/research/data/s2-NAME by driving the
# virtual Microduck with a pilot script (tools/session.py), the driver's recorder on
# (ITO_MICRODUCK_RECORD, with the sim's truth poses and every 3rd frame's depth, realistic ToF).
# Run in WSL on Ceres. Takes ~/.unii/work/ceres-session.lock; refuses from 19:55 and while
# USER-ON-CERES exists. Environment:
#   REC_REF  the driver (and session.py) to record with, a git ref snapshotted with git archive:
#            default 8fb1efb (lane base's A1 capture stamps; the recorder writes `captured`)
#   BUILD    the pilot build that drives (default feat-mast3r-slam-full; needs ITO_PILOT_SCRIPT)
#   PREFIX   the sim install (default SteadyNoodle's prefix: the shared install + OpenCV for the
#            beta fisheye); the shared ~/.local/share/ito-microduck is never changed
set -u
name=$1; script=$(readlink -f "$2")
REPO=$HOME/Projects/ito/main
REC_REF=${REC_REF:-8fb1efb}
BUILD=${BUILD:-feat-mast3r-slam-full}
PREFIX=${PREFIX:-$HOME/.unii/work/SteadyNoodle/sim-prefix}
LOCK=$HOME/.unii/work/ceres-session.lock
tape=$HOME/Projects/ito/research/data/s2-$name
session=tapes-$name
[ -e "$tape" ] && { echo "$tape exists"; exit 2; }
sha=$(git -C "$REPO" rev-parse --verify -q "$REC_REF^{commit}") || { echo "unknown ref $REC_REF"; exit 2; }
code=$HOME/Projects/ito/out/tapes/code/$sha
if [ ! -d "$code" ]; then
    mkdir -p "$code.tmp" && git -C "$REPO" archive "$sha" | tar -x -C "$code.tmp" && mv "$code.tmp" "$code"
fi
late() { [ "$(date +%H%M)" -ge 1955 ] || [ "$(date +%H%M)" -lt 0700 ]; }
export name script tape session code BUILD PREFIX
while true; do
    late && { echo "$(date +%T) not starting: Ceres is the user's from 19:55"; exit 75; }
    flock "$LOCK" bash -c '
        [ -e "$HOME/.unii/work/USER-ON-CERES" ] && exit 75
        ss -ltn | grep -q ":8081 " && exit 75
        n=$(cd /mnt/c && powershell.exe -NoProfile -Command "(Get-Process -ErrorAction SilentlyContinue | ? {\$_.ProcessName -match \"^(ito)$\"}).Count" 2>/dev/null < /dev/null | tr -d "\r ")
        [ "${n:-0}" != "0" ] && exit 75
        rm -rf "/mnt/c/Users/me/Projects/ito/out/sessions/$session"
        echo "$(date +%T) recording $tape"
        cd "$code" || exit 1
        ITO_MICRODUCK_RECORD=$tape ITO_MICRODUCK_TRUTH_DEPTH=3 \
        MICRODUCK_TOF_FOOTPRINT=5 MICRODUCK_TOF_UNMEASURED=0.25 \
            timeout 600 "$HOME/.local/bin/uv" run -q tools/session.py "$BUILD" "$session" --script "$script" \
            --sim-source "$code" --sim-prefix "$PREFIX" > "$HOME/Projects/ito/out/tapes/logs/rec-$name.log" 2>&1 < /dev/null
        rc=$?; [ $rc = 75 ] && rc=76; exit $rc'
    rc=$?
    [ $rc = 75 ] && { echo "$(date +%T) Ceres busy, waiting"; sleep 30; continue; }
    break
done
echo "$(date +%T) rc=$rc $(du -sh "$tape" 2>/dev/null | cut -f1) $(wc -l < "$tape/frames.jsonl" 2>/dev/null) frames"
echo "$REC_REF = $sha, build $BUILD, script $(basename "$script")" > "$tape/source.txt" 2>/dev/null
cp "$script" "$tape/script.json" 2>/dev/null
exit $rc
