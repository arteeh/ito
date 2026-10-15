#!/usr/bin/env bash
# Freeze new causal estimates, then measure the existing mapper before any tuning.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/recon-40-41/mapper/before}"
original="$HOME/Projects/ito/out/oracle-40"
frozen="${2:-}"
if [[ -n "$frozen" ]]; then shift 2; elif (( $# )); then shift; fi
extra=("$@")
runtime=/mnt/c/Users/me/Projects/ito/builds/feat-recon-40-41/ito/runtime/python.exe
mkdir -p "$output/visual"
for sequence in sweep180 tour; do
    mkdir -p "$output/$sequence"
    if [[ -n "$frozen" ]]; then
        [[ -e "$output/$sequence/causal-full" ]] || ln -s "$frozen/$sequence/causal-full" "$output/$sequence/causal-full"
    else
        mkdir -p "$output/$sequence/causal-full"
    fi
    for directory in source received; do
        [[ -e "$output/$sequence/$directory" ]] || ln -s "$original/$sequence/$directory" "$output/$sequence/$directory"
    done
    if [[ -z "$frozen" ]]; then
    echo "Freezing $sequence"
    flock /mnt/c/Users/me/Projects/ito/ceres.lock "$runtime" -I \
        "$(wslpath -w "$PWD/e2e/oracle_estimate.py")" "$(wslpath -w "$original/$sequence")" \
        --models 'C:\Users\me\Projects\ito\models-v5' \
        --out "$(wslpath -w "$output/$sequence/causal-full")" < /dev/null \
        > "$output/estimate-$sequence.log" 2>&1
    fi
    duration=17
    if [[ "$sequence" == tour ]]; then duration=85; fi
    for arm in ${ITO_ORACLE_ARMS:-TT TE ET EE TF main-TT main-TE main-ET main-EE}; do
        if [[ "${ITO_ORACLE_RESUME:-0}" == 1 && -f "$output/visual/$sequence-$arm/worker.json" && -f "$output/visual/$sequence-$arm/display.json" ]]; then
            .venv/bin/python - "$output/visual/$sequence-$arm/display.json" "$output/$sequence/causal-full/frames.json" <<'VERIFY'
import hashlib,json,subprocess,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text())["settings"]
assert s["code_commit"] == subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(), "Commit changed: use a fresh cohort output"
assert s["estimate_schedule_sha256"] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(), "Frozen schedule changed"
VERIFY
            echo "Reusing $sequence $arm"
            continue
        fi
        echo "Mapping $sequence $arm"
        bash e2e/oracle.sh e2e/oracle_run.py "$output/$sequence" --arm "$arm" \
            --out "$output/visual/$sequence-$arm" --seconds "$duration" --record "${extra[@]}" \
            > "$output/$sequence-$arm.log" 2>&1
    done
done
