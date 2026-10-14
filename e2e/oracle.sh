#!/usr/bin/env bash
# All CUDA/GL experiment commands use the same compiled kernel environment and device lock.
set -euo pipefail
cd "$(dirname "$0")/.."
output="$HOME/Projects/ito/out/oracle-40"
export DISPLAY=:0 GALLIUM_DRIVER=d3d12 PYTHONPATH="$PWD" PYTHONUTF8=1
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$output/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for library in "$PWD"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$library:$LD_LIBRARY_PATH"
done
export PATH="$PWD/.venv/bin:$PATH"
export CUDA_HOME="$output/cuda"
export CXX="$output/gcc/root/usr/bin/g++-13" CC="$output/gcc/root/usr/bin/gcc-13"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
exec flock /mnt/c/Users/me/Projects/ito/ceres.lock .venv/bin/python "$@"
