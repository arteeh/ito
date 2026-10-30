#!/bin/bash
# The user's virtual Microduck with the driver's recorder on: every drive becomes a tape
# ~/Projects/ito/research/data/user-MMDD-HHMM (frames, the sim's truth poses and depth, motion,
# head IMU, ToF), the same kind tools/tapes/record.sh makes. Installed as
# C:\Users\me\Projects\ito\sprint2\sim-record.sh and started by "0 Start sim + record.cmd".
# The driver is a snapshot of REC_SHA (lane base's A1 capture stamps), so the pilot must know
# them too: "1 Ito (for recording).cmd" starts the build of that commit (builds\tapes-rec-a1).
# The shared sim install is used as is (never changed): SteadyNoodle's prefix links to it and
# adds OpenCV for the beta fisheye.
set -u
REC_SHA=${REC_SHA:-8fb1efb19c128c803b7de2059ae2591dddbe7342}
code=$HOME/Projects/ito/out/tapes/code/$REC_SHA
prefix=$HOME/.unii/work/SteadyNoodle/sim-prefix
if [ ! -d "$code" ]; then
    mkdir -p "$code.tmp" && git -C "$HOME/Projects/ito/main" archive "$REC_SHA" | tar -x -C "$code.tmp" \
        && mv "$code.tmp" "$code" || { echo "could not unpack the driver $REC_SHA"; exit 1; }
fi
tape=$HOME/Projects/ito/research/data/user-$(date +%m%d-%H%M)
free=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
echo "Recording this drive to $tape (about 1 GB per minute; $free GB free)."
echo "Close this window or press Ctrl+C to stop the sim and the recording."
cd "$code" || exit 1
ITO_MICRODUCK_RECORD=$tape ITO_MICRODUCK_TRUTH_DEPTH=3 \
MICRODUCK_TOF_FOOTPRINT=5 MICRODUCK_TOF_UNMEASURED=0.25 \
PYTHONPATH=$code GST_PLUGIN_PATH=$prefix/gst-plugins-rs/target/debug \
    exec "$prefix/ito-venv/bin/python" -m drivers.microduck.sim \
    --microduck "$prefix/microduck" --rl "$prefix/microduck_rl" --policies "$prefix/policies/current" \
    --no-idle --viewer --gl glfw
