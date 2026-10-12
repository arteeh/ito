#!/bin/bash
set -e
cd /mnt/c/Users/me/Projects/ito/work/gs
while kill -0 17477 2>/dev/null; do sleep 5; done
scripts/run_env.sh scripts/train.py tour --init sparse --name true-sparse --seconds 180 > out/train-tour-sparse.log 2>&1
scripts/run_env.sh scripts/train.py tour --init calibrated --name true-calibrated --seconds 180 > out/train-tour-calibrated.log 2>&1
scripts/run_env.sh scripts/train.py tour --pose fused --init calibrated --name fused-calibrated --seconds 180 > out/train-tour-fused-calibrated.log 2>&1
../../venv-slam/Scripts/python.exe -B scripts/baseline_holdout.py sweep180 > out/baseline-heldout-sweep.log 2>&1
../../venv-slam/Scripts/python.exe -B scripts/baseline_holdout.py tour > out/baseline-heldout-tour.log 2>&1
scripts/run_env.sh scripts/baseline_eval.py sweep180 > out/baseline-eval-sweep.log 2>&1
scripts/run_env.sh scripts/baseline_eval.py tour > out/baseline-eval-tour.log 2>&1
