#!/bin/bash
cd /mnt/c/Users/me/Projects/ito/work/gs
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_CACHE_PATH="$PWD/cache/cuda-linux"
export TORCH_HOME="$PWD/cache/torch"
export XDG_CACHE_HOME="$PWD/cache"
export PYTHONDONTWRITEBYTECODE=1
exec .venv-gs/bin/python "$@"
