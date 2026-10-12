#!/bin/bash
# GPU queue after the tour init runs: online clouds, causal streams, pose split, renderer check,
# fused tour training. One GPU job at a time, all at nice 19.
cd /mnt/c/Users/me/Projects/ito/work/gs
while systemctl --user is-active -q gs-q1; do sleep 10; done
WIN=../../venv-slam/Scripts/python.exe
run() { local log=$1; shift; echo "START $log $(date +%T)"; nice -n 19 "$@" > out/$log.log 2>&1; echo "END $log rc=$? $(date +%T)"; }
run extract-sweep-online $WIN -B scripts/extract.py sweep180 online
run extract-tour-online $WIN -B scripts/extract.py tour online
run stream-sweep-true scripts/run_env.sh scripts/stream.py sweep180
run stream-tour-true-calibrated scripts/run_env.sh scripts/stream.py tour --init calibrated
run stream-tour-true scripts/run_env.sh scripts/stream.py tour
run stream-sweep-fused-online scripts/run_env.sh scripts/stream.py sweep180 --pose fused --init online
run stream-tour-fused-online scripts/run_env.sh scripts/stream.py tour --pose fused --init online
run evalpose-sweep-fused scripts/run_env.sh scripts/eval_pose.py sweep180 fused
run evalpose-sweep-true15 scripts/run_env.sh scripts/eval_pose.py sweep180 true-30/checkpoint-15s --pose true
run reference scripts/run_env.sh scripts/reference.py sweep180 true-30/checkpoint-15s
run ito-match $WIN -B scripts/ito_match.py sweep180 true-30/checkpoint-15s
run train-tour-fused-calibrated scripts/run_env.sh scripts/train.py tour --pose fused --init calibrated --name fused-calibrated --seconds 180
run evalpose-tour-fused-calibrated scripts/run_env.sh scripts/eval_pose.py tour fused-calibrated --init calibrated
run train-tour-fused-online scripts/run_env.sh scripts/train.py tour --pose fused --init online --name fused-online --seconds 60
run train-tour-true-online scripts/run_env.sh scripts/train.py tour --init online --name true-online --seconds 60
echo QUEUE-DONE
