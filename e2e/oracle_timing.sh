#!/usr/bin/env bash
# No Recorder or image readback; three repeats of the production GL/CUDA workload.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/oracle-40}"
for repeat in 1 2 3; do
    for sequence in sweep180 tour; do
        duration=17
        if [[ "$sequence" == tour ]]; then duration=85; fi
        for arm in main-TT TT; do
            name="$sequence-$arm-$repeat"
            bash e2e/oracle.sh e2e/oracle_run.py "$output/$sequence" --arm "$arm" \
                --out "$output/timing/$name" --seconds "$duration" \
                > "$output/$name-timing.log" 2>&1
        done
    done
done
