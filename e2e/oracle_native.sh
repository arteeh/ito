#!/usr/bin/env bash
# Native Windows NVIDIA GL + the actual WSL mapper; all children inside one Ceres lock.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/oracle-40}"
python=/mnt/c/Users/me/Projects/ito/builds/main/ito/runtime/python.exe
for repeat in 1 2 3; do
    for sequence in sweep180 tour; do
        for mode in main-TT TT-frozen TT TF-frozen TF; do
            name="$sequence-$mode-$repeat"
            arm="$mode"
            if [[ "$mode" == *-frozen ]]; then arm="${mode%-frozen}"; fi
            load=()
            if [[ "$mode" == TT || "$mode" == TF ]]; then load=(--load); fi
            echo "Native GPU countercheck: $name"
            flock /mnt/c/Users/me/Projects/ito/ceres.lock "$python" -I \
                "$(wslpath -w "$PWD/e2e/oracle_native.py")" \
                "$(wslpath -w "$output/$sequence")" \
                --out "$(wslpath -w "$output/native/$name")" \
                --linux-root "$PWD" --arm "$arm" "${load[@]}" --seconds 20 < /dev/null \
                > "$output/native-$name.log" 2>&1
        done
    done
done
