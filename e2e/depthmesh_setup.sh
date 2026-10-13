#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(realpath -m ../p47-out)"
mkdir -p "$root/weights" "$root/cache/torch"
uv venv "$root/venv" --python 3.12 --allow-existing
# Install Torch first so dependencies cannot pull a different CUDA release.
uv pip install --python "$root/venv/bin/python" torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$root/venv/bin/python" -e . pillow scipy scikit-image \
    imageio imageio-ffmpeg opencv-python-headless 'mujoco==3.3.7' ruff einops \
    huggingface-hub timm matplotlib lpips
clone() {
    if [[ ! -d "$root/$1/.git" ]]; then
        git clone "$2" "$root/$1"
        git -C "$root/$1" checkout "$3"
    fi
    test "$(git -C "$root/$1" rev-parse HEAD)" = "$3"
}
clone da2 https://github.com/DepthAnything/Depth-Anything-V2.git a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
clone vda https://github.com/DepthAnything/Video-Depth-Anything.git 4f5ae23172ba60fd7bc11ef671cca678842c7072
clone promptda https://github.com/DepthAnything/PromptDA.git 9161547ca7151e14fcc8f490ad3cc02fa9437483
fetch() {
    if [[ ! -f "$root/weights/$1" ]]; then
        curl -fL --retry 3 "https://huggingface.co/depth-anything/$2/resolve/main/$3" \
            -o "$root/weights/$1.part"
        mv "$root/weights/$1.part" "$root/weights/$1"
    fi
}
fetch da2-small.pth Depth-Anything-V2-Small depth_anything_v2_vits.pth
fetch da2-large.pth Depth-Anything-V2-Large depth_anything_v2_vitl.pth
fetch metric-small.pth Depth-Anything-V2-Metric-Hypersim-Small depth_anything_v2_metric_hypersim_vits.pth
fetch metric-large.pth Depth-Anything-V2-Metric-Hypersim-Large depth_anything_v2_metric_hypersim_vitl.pth
fetch video-small.pth Video-Depth-Anything-Small video_depth_anything_vits.pth
fetch prompt-small.ckpt prompt-depth-anything-vits model.ckpt
uv pip freeze --python "$root/venv/bin/python" > "$root/packages.txt"
echo "Isolated environment and weights ready: $root"
