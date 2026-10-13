#!/usr/bin/env bash
# Hidden offscreen Windows GL only; release the GPU between bounded jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(realpath -m ../p48-out)"
mkdir -p "$root/logs"
export PYTHONPATH="$(wslpath -w "$PWD")" PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export WSLENV=PYTHONPATH:PYTHONUTF8:PYTHONDONTWRITEBYTECODE
exec timeout --signal=TERM --kill-after=5 1100 \
    flock /mnt/c/Users/me/Projects/ito/ceres.lock \
    "$root/winvenv/Scripts/python.exe" "$@"
