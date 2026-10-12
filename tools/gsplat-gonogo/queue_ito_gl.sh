#!/bin/bash
# Ito renderer match + Ito GL stereo speed for a small, a 440k and a 1.1M Gaussian model.
cd /mnt/c/Users/me/Projects/ito/work/gs
while systemctl --user is-active -q gs-q2 || systemctl --user is-active -q gs-q3 || systemctl --user is-active -q gs-q4; do sleep 10; done
WIN=../../venv-slam/Scripts/python.exe
run() { local log=$1; shift; echo "START $log $(date +%T)"; nice -n 19 "$@" > out/$log.log 2>&1; echo "END $log rc=$? $(date +%T)"; }
run ito-match $WIN -B scripts/ito_match.py sweep180 true-30/checkpoint-15s
run reference-sweep-true scripts/run_env.sh scripts/reference.py sweep180 true
run ito-match-sweep-true $WIN -B scripts/ito_match.py sweep180 true
run reference-tour-cal30 scripts/run_env.sh scripts/reference.py tour true-calibrated/checkpoint-30s
run ito-match-tour-cal30 $WIN -B scripts/ito_match.py tour true-calibrated/checkpoint-30s
