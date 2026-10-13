#!/usr/bin/env bash
# One command: isolated/probe/reference/visual traces, then the frozen quality replay.
set -euo pipefail
cd "$(dirname "$0")/.."
work="$(dirname "$PWD")"
out="${1:-$work/m2-out/verified}"
mkdir -p "$out"
run() {
    local name="$1"
    shift
    "$work/m2-out/venv/bin/python" - "$@" > "$out/$name-request.json" <<'PYTHON'
import json, sys
sys.path.insert(0, "e2e")
from gpu_budget import fingerprint
print(json.dumps(dict(source_fingerprint=fingerprint(), arguments=sys.argv[1:])))
PYTHON
    if [[ "${ITO_GPU_RESUME:-1}" == 1 && -f "$out/$name/display.json" ]] && \
        cmp -s "$out/$name-request.json" "$out/$name-complete.json"; then
        echo "Reusing exact completed GPU case: $name"
        return
    fi
    echo "GPU budget: $name"
    bash e2e/gpu_budget.sh "$out/$name" "$@" > "$out/$name.log" 2>&1
    cp "$out/$name-request.json" "$out/$name-complete.json"
}
ITO_GPU_ENTRY=e2e/gpu_matching.py run matching-fuzz
ITO_GPU_ENTRY=e2e/gpu_solver_fuzz.py run solver-fuzz --load slam --admit --frames 2700 --warm-prefix 1560
for count in 131072 262144 524288; do
    run "occupied-$count" --occupied "$count" --capacity 524288 --incremental --frames 900
done
for capacity in 524288 1048576; do
    run "reserved-$capacity" --occupied 262144 --capacity "$capacity" --incremental --frames 900
    run "full-sort-$capacity" --occupied 262144 --capacity "$capacity" --frames 900
done
run unregulated --load both --frames 1800 --incremental --updates
run trace --load slam --frames 900 --profile --warm-prefix 1560
run trace-admitted --load slam --frames 2700 --profile --warm-prefix 1560 --admit
for count in 131072 262144 524288; do
    frames=900
    if [[ "$count" == 524288 ]]; then frames=6000; fi
    run "loaded-occupied-$count" --sequence tour --occupied "$count" --capacity 524288 \
        --incremental --updates --load both --admit --frames "$frames" --warm-prefix 1560 \
        --eye-size 2160x2160 --target-format f1
done
for repeat in 1 2 3; do
    for sequence in sweep180 tour; do
        prefix=12
        if [[ "$sequence" == tour ]]; then prefix=1560; fi
        run "$sequence-$repeat" --sequence "$sequence" --load both --frames 6000 \
            --admit --incremental --updates --capacity 524288 --warm-prefix "$prefix"
        run "$sequence-full-$repeat" --sequence "$sequence" --load both --frames 6000 \
            --admit --incremental --updates --capacity 524288 --warm-prefix "$prefix" \
            --eye-size 2160x2160 --target-format f1
   done
done
run visual-tour --sequence tour --load both --frames 900 --admit --incremental --updates \
    --visual --head-offset 48 --view-index 1500
run visual-sweep --sequence sweep180 --load both --frames 900 --admit --incremental --updates \
    --visual --head-offset 12 --view-index 452
ITO_MEASURE_CANDIDATE=1 ITO_MEASURE_BUDGET=524288 ITO_MEASURE_RESUME=1 \
    ITO_MEASURE_CACHE="$work/m2-out" \
    bash e2e/measure_baseline.sh "$out/baseline" > "$out/baseline.log" 2>&1
"$work/m2-out/venv/bin/python" e2e/gpu_budget_report.py "$out"
