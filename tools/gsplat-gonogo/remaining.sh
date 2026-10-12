#!/bin/bash
set -e
cd /mnt/c/Users/me/Projects/ito/work/gs
scripts/run_env.sh scripts/train.py tour --seconds 180 > out/train-tour-true.log 2>&1
scripts/run_env.sh scripts/train.py sweep180 --pose fused --seconds 180 > out/train-sweep-fused.log 2>&1
scripts/run_env.sh scripts/train.py tour --pose fused --seconds 180 > out/train-tour-fused.log 2>&1
scripts/run_env.sh scripts/train.py sweep180 --name true-30 --seconds 30 > out/train-sweep-30.log 2>&1
../../venv-slam/Scripts/python.exe -B scripts/baseline.py sweep180 > out/baseline-sweep.log 2>&1
../../venv-slam/Scripts/python.exe -B scripts/baseline.py tour > out/baseline-tour.log 2>&1
scripts/run_env.sh scripts/stream.py sweep180 > out/stream-sweep-true.log 2>&1
scripts/run_env.sh scripts/stream.py tour > out/stream-tour-true.log 2>&1
scripts/run_env.sh scripts/stream.py sweep180 --pose fused > out/stream-sweep-fused.log 2>&1
scripts/run_env.sh scripts/stream.py tour --pose fused > out/stream-tour-fused.log 2>&1
scripts/run_env.sh scripts/reference.py sweep180 > out/reference.log 2>&1
../../venv-slam/Scripts/python.exe -B scripts/ito_match.py sweep180 > out/ito-match.log 2>&1
