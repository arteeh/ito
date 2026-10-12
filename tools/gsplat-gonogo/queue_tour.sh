#!/bin/bash
# GPU queue: tour training with sparse/calibrated init (one GPU job at a time).
cd /mnt/c/Users/me/Projects/ito/work/gs
for spec in "true sparse true-sparse" "true calibrated true-calibrated"; do
  set -- $spec
  echo "START $3 $(date +%T)"
  nice -n 19 scripts/run_env.sh scripts/train.py tour --pose $1 --init $2 --name $3 --seconds 180 > out/train-tour-$3.log 2>&1
  echo "END $3 rc=$? $(date +%T)"
done
