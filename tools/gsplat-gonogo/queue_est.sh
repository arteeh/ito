#!/bin/bash
# Fused tour on estimated frames only (no interpolation through head turns), then its pose split.
cd /mnt/c/Users/me/Projects/ito/work/gs
while systemctl --user is-active -q gs-q2 || systemctl --user is-active -q gs-q3 || systemctl --user is-active -q gs-q4 || systemctl --user is-active -q gs-q5; do sleep 10; done
run() { local log=$1; shift; echo "START $log $(date +%T)"; nice -n 19 "$@" > out/$log.log 2>&1; echo "END $log rc=$? $(date +%T)"; }
run train-tour-fused-cal-est scripts/run_env.sh scripts/train.py tour --pose fused --init calibrated --estimated-only --name fused-calibrated-est --seconds 60
run train-tour-true-cal-est scripts/run_env.sh scripts/train.py tour --init calibrated --estimated-only --name true-calibrated-est --seconds 60
run evalpose-tour-fused-cal-est scripts/run_env.sh scripts/eval_pose.py tour fused-calibrated-est --init calibrated --estimated-only
