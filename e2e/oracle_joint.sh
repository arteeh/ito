#!/usr/bin/env bash
# Freeze new causal estimates, then measure the existing mapper before any tuning.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/recon-40-41/mapper/before}"
original="$HOME/Projects/ito/out/oracle-40"
runtime=/mnt/c/Users/me/Projects/ito/builds/feat-recon-40-41/ito/runtime/python.exe
mkdir -p "$output/visual"
for sequence in sweep180 tour; do
    mkdir -p "$output/$sequence/causal-full"
    for directory in source received; do
        [[ -e "$output/$sequence/$directory" ]] || ln -s "$original/$sequence/$directory" "$output/$sequence/$directory"
    done
    echo "Freezing $sequence"
    flock /mnt/c/Users/me/Projects/ito/ceres.lock "$runtime" -I \
        "$(wslpath -w "$PWD/e2e/oracle_estimate.py")" "$(wslpath -w "$original/$sequence")" \
        --models 'C:\Users\me\Projects\ito\models-v5' \
        --out "$(wslpath -w "$output/$sequence/causal-full")" < /dev/null \
        > "$output/estimate-$sequence.log" 2>&1
    duration=17
    if [[ "$sequence" == tour ]]; then duration=85; fi
    for arm in TT TE ET EE TF main-TT main-TE main-ET main-EE; do
        echo "Before $sequence $arm"
        bash e2e/oracle.sh e2e/oracle_run.py "$output/$sequence" --arm "$arm" \
            --out "$output/visual/$sequence-$arm" --seconds "$duration" --record \
            > "$output/$sequence-$arm.log" 2>&1
    done
done
