#!/usr/bin/env bash
# Machine-local offline baseline: CPU/EGL images, serialized Windows CUDA replays.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
work="$(dirname "$root")"
output="${1:-$work/m1-out/baseline-final}"
mkdir -p "$output"
output="$(realpath "$output")"
python="$work/m1-out/venv/bin/python"
slam_python="${ITO_MEASURE_SLAM_PYTHON:-$work/../venv-slam/Scripts/python.exe}"
models="${ITO_MEASURE_MODELS:-$work/pp/models}"
if [[ ! -x "$python" || ! -f "$slam_python" ]]; then
    echo 'Measurement environments missing; see docs/measure.md' >&2
    exit 1
fi
# Refuse to label modified product code as the pinned main baseline.
git diff --exit-code HEAD -- ito drivers > "$output/product-diff.txt"
"$python" -c 'import json, subprocess; from pathlib import Path; r=json.loads(Path("e2e/measure_reference.json").read_text()); assert all(subprocess.check_output(["git", "rev-parse", "HEAD:"+p], text=True).strip() == r[p+"_tree"] for p in ("ito", "drivers")), "Product differs from pinned main"'
export PYTHONPATH="$root" PYTHONDONTWRITEBYTECODE=1 LIBGL_ALWAYS_SOFTWARE=1
export TORCH_HOME="$work/m1-out/torch"
"$python" -m pip --version >/dev/null 2>&1 && "$python" -m pip freeze > "$output/packages.txt" || uv pip freeze --python "$python" > "$output/packages.txt"
"$python" e2e/measure.py --work "$work" --out "$output" --prepare
"$python" e2e/measure_source.py --work "$work" --out "$output/source" > "$output/source.log" 2>&1
for repeat in 1 2 3; do
    run_out="$output/repeat-$repeat"
    mkdir -p "$run_out"
    cp "$output/inputs.json" "$output/"*-schedule.json "$run_out/"
    for sequence in sweep180 tour; do
        echo "SLAM repeat $repeat: $sequence"
        env PYTHONPATH="$(wslpath -w "$root")" PYTHONUTF8=1 \
            CUPY_CACHE_DIR="$(wslpath -w "$work/m1-out/cupy-cache")" \
            WSLENV=PYTHONPATH:PYTHONUTF8:PYTHONDONTWRITEBYTECODE:CUPY_CACHE_DIR \
            flock "$work/../ceres.lock" timeout 1100 "$slam_python" \
            e2e/measure_capture.py "$(wslpath -w "$work/rf-run/out/$sequence/rec")" \
            --schedule "$(wslpath -w "$output/$sequence-schedule.json")" \
            --models "$(wslpath -w "$models")" \
            --out "$(wslpath -w "$run_out/$sequence")" > "$run_out/$sequence.log" 2>&1
    done
    "$python" e2e/measure.py --work "$work" --out "$run_out" --ipd "${ITO_MEASURE_IPD:-0.064}" > "$run_out/images.log" 2>&1
done
for repeat in 1 2 3; do
    env PYTHONPATH="$(wslpath -w "$root")" PYTHONUTF8=1 \
        WSLENV=PYTHONPATH:PYTHONUTF8:PYTHONDONTWRITEBYTECODE \
        flock "$work/../ceres.lock" timeout 1100 "$slam_python" e2e/measure_display.py \
        "$(wslpath -w "$output/repeat-$repeat")" --work "$(wslpath -w "$work")" \
        --ipd "${ITO_MEASURE_IPD:-0.064}" > "$output/repeat-$repeat/display.log" 2>&1
done
"$python" e2e/measure_audit.py "$output"
echo "Baseline complete: $output"
