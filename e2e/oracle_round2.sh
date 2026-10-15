#!/usr/bin/env bash
# Reuse the exact round-one observations; leave its evidence intact.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-$HOME/Projects/ito/out/oracle-40/round2}"
mkdir -p "$output/visual" "$output/native" "$output/review"
for sequence in sweep180 tour; do
    mkdir -p "$output/$sequence"
    for input in source causal-full; do
        if [[ ! -e "$output/$sequence/$input" ]]; then
            ln -s "../../$sequence/$input" "$output/$sequence/$input"
        fi
    done
    duration=17
    if [[ "$sequence" == tour ]]; then duration=85; fi
    for arm in TT TF main-TT TE main-TE ET main-ET EE main-EE; do
        name="$sequence-$arm"
        if [[ "${2:-}" == --reuse-controls && "$arm" == main-* && -f "$output/visual/$name/worker.json" && -f "$output/visual/$name/display.json" ]]; then
            echo "Reusing completed main control: $name"
            continue
        fi
        echo "Round 2 visual: $name"
        bash e2e/oracle.sh e2e/oracle_run.py "$output/$sequence" --arm "$arm" \
            --out "$output/visual/$name" --seconds "$duration" --record \
            > "$output/$name.log" 2>&1
    done
done
# Timing runs precede CPU snapshot rendering to avoid contention from report generation.
bash e2e/oracle_native.sh "$output"
for sequence in sweep180 tour; do
    for arm in TT TE ET EE TF; do
        PYTHONPATH=. LP_NUM_THREADS=4 .venv/bin/python e2e/oracle_report.py \
            --out "$output" --sequence "$sequence" --arm "$arm" \
            --matched-controls --previous "$(dirname "$output")" --images "$output/review" > "$output/report-$sequence-$arm.log" 2>&1
    done
done
PYTHONPATH=. .venv/bin/python e2e/oracle_audit.py "$output"
