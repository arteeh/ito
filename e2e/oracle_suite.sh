#!/usr/bin/env bash
# Each arm gets its own process, fresh optimizer, and the same frozen input schedule.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/oracle-40}"
for sequence in sweep180 tour; do
    duration=17
    if [[ "$sequence" == tour ]]; then duration=85; fi
    for arm in main-TT TT TE ET EE TF main-EE; do
        name="$sequence-$arm"
        echo "Oracle: $name"
        bash e2e/oracle.sh e2e/oracle_run.py "$output/$sequence" --arm "$arm" \
            --out "$output/visual/$name" --seconds "$duration" --record \
            > "$output/$name-visual.log" 2>&1
    done
done
