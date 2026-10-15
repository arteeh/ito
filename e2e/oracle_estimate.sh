#!/usr/bin/env bash
# The Windows release supplies Python/native CUDA dependencies; source is always this WSL tree.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/oracle-40}"
release=/mnt/c/Users/me/Projects/ito/builds/feat-recon-40-41/ito
for sequence in sweep180 tour; do
    flock /mnt/c/Users/me/Projects/ito/ceres.lock "$release/runtime/python.exe" -I \
        "$(wslpath -w "$PWD/e2e/oracle_estimate.py")" "$(wslpath -w "$output/$sequence")" \
        --models 'C:\Users\me\Projects\ito\models-v5' \
        --out "$(wslpath -w "$output/$sequence/causal-full")" < /dev/null \
        > "$output/causal-$sequence.log" 2>&1
done
