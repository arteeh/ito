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
# Separate native image captures, after all timing and CPU report work have finished.
native_python=/mnt/c/Users/me/Projects/ito/builds/main/ito/runtime/python.exe
for sequence in sweep180 tour; do
    late=12
    if [[ "$sequence" == tour ]]; then late=78; fi
    for moment in 1 "$late"; do
        for arm in main-TT TT TF; do
            name="$sequence-$arm-$(.venv/bin/python -c 'import sys; print(f"{float(sys.argv[1]):.1f}")' "$moment")"
            flock /mnt/c/Users/me/Projects/ito/ceres.lock "$native_python" -I \
                "$(wslpath -w "$PWD/e2e/oracle_native.py")" \
                "$(wslpath -w "$output/$sequence")" \
                --out "$(wslpath -w "$output/native-views/$name")" \
                --linux-root "$PWD" --arm "$arm" --moment "$moment" \
                --seconds 0.1 --capture < /dev/null > "$output/native-view-$name.log" 2>&1
        done
        PYTHONPATH=.:e2e .venv/bin/python - "$output" "$sequence" "$moment" <<'SHEET'
import sys
from pathlib import Path
from oracle_report import native_views
out=Path(sys.argv[1]); seq=sys.argv[2]; moment=float(sys.argv[3])
native_views(out,seq,moment,out/"review"/f"{seq}-native-{moment:.1f}.jpg")
SHEET
    done
done
