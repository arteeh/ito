#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(realpath -m ../p48-out)"
mkdir -p "$root/cache/torch/hub/checkpoints" "$root/logs"
uv venv "$root/venv" --python 3.12 --allow-existing
uv pip install --python "$root/venv/bin/python" -e . pillow scipy scikit-image \
    imageio imageio-ffmpeg 'opencv-python-headless==4.11.0.86' 'mujoco==3.3.7' ruff
uv pip install --python "$root/venv/bin/python" torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$root/venv/bin/python" lpips
# Windows GL dependencies are read only; this venv and its path file are private.
windows_python="$(realpath ../../venv-slam/Scripts/python.exe)"
"$windows_python" -m venv "$(wslpath -w "$root/winvenv")"
printf '%s\n' "$(wslpath -m "$(realpath ../../venv-slam/Lib/site-packages)")" \
    > "$root/winvenv/Lib/site-packages/baseline-readonly.pth"
uv pip freeze --python "$root/venv/bin/python" > "$root/packages.txt"
