#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
work="$(dirname "$root")"
output="${1:-$work/m2-out/trace}"
shift || true
mkdir -p "$output"
export PYTHONPATH="$(wslpath -w "$root")" PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="$(wslpath -w "$work/m2-out/cupy-cache")"
export WSLENV=PYTHONPATH:PYTHONUTF8:PYTHONDONTWRITEBYTECODE:CUPY_CACHE_DIR
flock "$work/../ceres.lock" timeout 1100 "$work/../venv-slam/Scripts/python.exe" \
    e2e/gpu_budget.py --work "$(wslpath -w "$work")" \
    --out "$(wslpath -w "$(realpath "$output")")" "$@"
