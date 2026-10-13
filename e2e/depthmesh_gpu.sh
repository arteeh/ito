#!/usr/bin/env bash
# The timeout also expires while waiting for the shared lock, before the quiet-GPU deadline.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(realpath ../p47-out)"
remaining=$(( $(TZ=Europe/Amsterdam date -d '2026-10-13 11:49:30' +%s) - $(date +%s) ))
if (( remaining <= 0 )); then
    echo 'Depth-mesh GPU deadline has passed; GPU work is disabled.' >&2
    exit 1
fi
limit=1100
if (( remaining < limit )); then limit=$remaining; fi
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_HOME="$root/cache/torch" HF_HOME="$root/cache/huggingface"
export CUDA_CACHE_PATH="$root/cache/cuda" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4
# inference uses Linux CUDA; all image evaluation forces Mesa in the other runner.
timeout --signal=TERM --kill-after=5 "$limit" flock /mnt/c/Users/me/Projects/ito/ceres.lock \
    "$root/venv/bin/python" e2e/depthmesh_infer.py "$@"
