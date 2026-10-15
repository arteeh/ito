#!/bin/sh
# Record a scripted pilot session in a Windows build of Ito against the virtual Microduck,
# then turn it into frame sheets to judge by eye (e2e/motion_sheet.py). Run on ceres, in WSL:
#
#   e2e/by_eye.sh BUILD NAME [ROUTE] [SIM_SOURCE]
#
# BUILD       a folder under C:\Users\me\Projects\ito\builds (release.ps1 -Branch ...)
# NAME        the run's name; output goes to C:\Users\me\Projects\ito\out\by-eye\NAME
# ROUTE       pilot script (ITO_PILOT_SCRIPT, see ito/desktop/input.py), default room-tour
# SIM_SOURCE  the checkout whose drivers/microduck runs the sim and driver, default this one
#
# VIEWER=1 also opens the MuJoCo viewer. The Ito window pops up and is driven by the script
# whatever has focus; nobody needs to touch it. The sim from setup-sim.sh must be installed.
set -eu
[ "$#" -ge 2 ] || { sed -n 2,13p "$0" >&2; exit 2; }
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build=$1 name=$2 route=${3:-$here/e2e/routes/room-tour.json} source=${4:-$here}
prefix=$HOME/.local/share/ito-microduck
root=/mnt/c/Users/me/Projects/ito
exe=$root/builds/$build/ito/ito.exe
out=$root/out/by-eye/$name
[ -x "$exe" ] || { echo "no build at $exe" >&2; exit 1; }
rm -rf "$out"
mkdir -p "$out"
cp "$route" "$out/route.json"
git -C "$source" log --oneline -1 > "$out/sim-source.txt"
cat "$root/builds/$build.log" > "$out/build.log" 2>/dev/null || true

sh "$root/stop-sim.sh" >/dev/null 2>&1 || true
PYTHONPATH=$source GST_PLUGIN_PATH=$prefix/gst-plugins-rs/target/debug \
    "$prefix/ito-venv/bin/python" -m drivers.microduck.sim \
    --microduck "$prefix/microduck" --rl "$prefix/microduck_rl" \
    --policies "$prefix/policies/current" ${VIEWER:+--viewer} > "$out/sim.log" 2>&1 &
sim=$!
trap 'sh "$root/stop-sim.sh" >/dev/null 2>&1 || true' EXIT
i=0
until grep -q "Pairing code" "$out/sim.log"; do
    i=$((i + 1))
    if [ $i -gt 180 ] || ! kill -0 $sim 2>/dev/null; then
        echo "sim did not start:" >&2; tail -20 "$out/sim.log" >&2; exit 1
    fi
    sleep 0.5
done
ip=$(hostname -I | cut -d' ' -f1)
code=$(cat "$HOME/.config/ito/microduck-sim-code")
wout=$(wslpath -w "$out")
echo "recording $build into $wout"
(cd "$root/builds/$build/ito" && cmd.exe /c "set ITO_RECORD=$wout\\rec&& set ITO_PILOT_SCRIPT=$wout\\route.json&& ito.exe $ip:8081 --code $code --capture-dir $wout\\captures" \
    > "$out/ito.log" 2>&1) || echo "ito exit $?" >> "$out/ito.log"
sh "$root/stop-sim.sh" >/dev/null 2>&1 || true
cp -r "$HOME/.local/state/ito/microduck-sim" "$out/sim-logs" 2>/dev/null || true
"$here/.venv/bin/python" "$here/e2e/motion_sheet.py" "$out/rec" --out "$out/sheets" > "$out/sheets.log" 2>&1 \
    || "$HOME/Projects/ito/main/.venv/bin/python" "$here/e2e/motion_sheet.py" "$out/rec" --out "$out/sheets" > "$out/sheets.log" 2>&1
echo "sheets: $out/sheets"
du -sh "$out"
