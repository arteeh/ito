#!/bin/bash
# Separate cloud scale from pose error in the causal tour test.
cd /mnt/c/Users/me/Projects/ito/work/gs
while systemctl --user is-active -q gs-q2; do sleep 10; done
run() { local log=$1; shift; echo "START $log $(date +%T)"; nice -n 19 "$@" > out/$log.log 2>&1; echo "END $log rc=$? $(date +%T)"; }
run stream-tour-fused-calibrated scripts/run_env.sh scripts/stream.py tour --pose fused --init calibrated
run stream-tour-true-online scripts/run_env.sh scripts/stream.py tour --init online
echo QUEUE-DONE
