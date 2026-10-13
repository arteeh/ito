#!/usr/bin/env bash
# Full offline experiment. --cached-only makes no CUDA/Windows GL calls.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(realpath -m ../p47-out)"
cached=0
if [[ "${1:-}" == --cached-only ]]; then cached=1; elif [[ $# != 0 ]]; then
    echo 'Usage: bash e2e/depthmesh_run.sh [--cached-only]' >&2; exit 2
fi
if [[ ! -x "$root/venv/bin/python" ]]; then
    if (( cached )); then echo 'Missing isolated environment; run depthmesh_setup.sh' >&2; exit 1; fi
    bash e2e/depthmesh_setup.sh
fi
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 LIBGL_ALWAYS_SOFTWARE=1
export TORCH_HOME="$root/cache/torch" OMP_NUM_THREADS=4
mkdir -p "$TORCH_HOME" "$root/logs"
python="$root/venv/bin/python"
"$python" e2e/depthmesh_prepare.py > "$root/logs/prepare.log" 2>&1
"$python" e2e/depthmesh_smoke.py
for variant in truth da2-small da2-large metric-small metric-large video-small prompt-small; do
    if [[ "$variant" != truth ]]; then
        for sequence in sweep180 tour; do
            report="$root/depth/$variant/$sequence/inference.json"
            if [[ ! -f "$report" ]]; then
                if (( cached )); then echo "Missing cached depth: $report" >&2; exit 1; fi
                # Remove the software renderer's env from inference; only CUDA uses the GPU here.
                env -u LIBGL_ALWAYS_SOFTWARE bash e2e/depthmesh_gpu.sh \
                    --variant "$variant" --sequence "$sequence" > "$root/logs/infer-$variant-$sequence.log" 2>&1
            fi
        done
    fi
    for pose in oracle robot; do
        echo "Rendering $variant / $pose"
        "$python" e2e/depthmesh_evaluate.py --variant "$variant" --pose "$pose" --save-world \
            > "$root/logs/render-$variant-$pose.log" 2>&1
    done
done
"$python" e2e/depthmesh_clips.py > "$root/logs/clips.log" 2>&1
"$python" e2e/depthmesh_offsets.py > "$root/logs/offsets.log" 2>&1
bash e2e/depthmesh_viewer.sh > "$root/logs/viewer.log" 2>&1
"$python" e2e/depthmesh_report.py
"$python" -m compileall -q ito/proto e2e/depthmesh_*.py
"$root/venv/bin/ruff" check ito/proto e2e/depthmesh_*.py
git diff --check
echo "Depth-mesh experiment complete: $root; GPU timings are separate archived runs."
