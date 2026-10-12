#!/bin/bash
# Control for eval_pose: refinement on the true-pose 180 s sweep model.
cd /mnt/c/Users/me/Projects/ito/work/gs
while systemctl --user is-active -q gs-q2 || systemctl --user is-active -q gs-q3; do sleep 10; done
echo "START evalpose-sweep-true $(date +%T)"
nice -n 19 scripts/run_env.sh scripts/eval_pose.py sweep180 true --pose true > out/evalpose-sweep-true.log 2>&1
echo "END evalpose-sweep-true rc=$? $(date +%T)"
