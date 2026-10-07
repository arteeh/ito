#!/bin/sh
# Ubuntu 24.04+: install the real Pollen stack under one prefix (apt needs root).
set -eu
[ "$#" = 1 ] || { echo "Usage: $0 PREFIX" >&2; exit 2; }
prefix=$(realpath -m "$1")
source=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$prefix"
if [ "$(id -u)" = 0 ]; then sudo_cmd=; else sudo_cmd=sudo; fi
$sudo_cmd apt-get -o Acquire::ForceIPv4=true update
$sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::ForceIPv4=true \
    install -y --no-install-recommends \
    build-essential pkg-config curl ca-certificates git python3-venv \
    libssl-dev libudev-dev libclang-dev clang libasound2-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-nice \
    libosmesa6 libgl1-mesa-dri libglfw3 libportaudio2
export UV_INSTALL_DIR="$prefix/tools" UV_NO_MODIFY_PATH=1
export UV_PYTHON_INSTALL_DIR="$prefix/python" UV_CACHE_DIR="$prefix/cache/uv"
export CARGO_HOME="$prefix/cargo" RUSTUP_HOME="$prefix/rustup"
export PATH="$prefix/tools:$CARGO_HOME/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -fsSL https://astral.sh/uv/install.sh -o "$prefix/uv-install.sh"
    sh "$prefix/uv-install.sh"
fi
if ! command -v rustup >/dev/null; then
    curl -fsSL https://sh.rustup.rs -o "$prefix/rust-install.sh"
    sh "$prefix/rust-install.sh" -y --no-modify-path --profile minimal --default-toolchain 1.95.0
fi
rustup toolchain install 1.95.0 --profile minimal
checkout() {
    dest=$1 url=$2 revision=$3
    if [ ! -d "$dest/.git" ]; then
        git init "$dest"
        git -C "$dest" remote add origin "$url"
    fi
    if ! git -C "$dest" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$dest" fetch --depth 1 origin "$revision"
        git -C "$dest" checkout --detach FETCH_HEAD
    fi
    [ "$(git -C "$dest" rev-parse HEAD)" = "$revision" ] || {
        echo "Unexpected revision in $dest; use a fresh prefix" >&2; exit 1;
    }
}
checkout "$prefix/microduck" https://github.com/pollen-robotics/microduck 77005fdeb330de64247fa4cc940ba90e2be3ea85
checkout "$prefix/microduck_rl" https://github.com/pollen-robotics/microduck_rl 273afe0b31c4ab365b9ff806a927b63ac92b5ddd
checkout "$prefix/gst-plugins-rs" https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs 18a88c81d1413562586a1e7458a3181a3bd31bd2
# The body server needs MuJoCo, not the RL training stack or CUDA/PyTorch.
uv venv --python 3.12 --allow-existing "$prefix/microduck_rl/.venv"
uv pip install --python "$prefix/microduck_rl/.venv/bin/python" \
    mujoco==3.3.7 numpy==2.2.6 onnxruntime==1.24.4 pillow==11.3.0
uv venv --python 3.12 --allow-existing "$prefix/ito-venv"
uv pip install --python "$prefix/ito-venv/bin/python" --no-cache \
    --reinstall-package ito-driver-microduck "$source/drivers/microduck"
nice -n 15 cargo +1.95.0 build --manifest-path "$prefix/microduck/Cargo.toml" \
    --locked -p robotd -p robotctl -p mediad -j 2
nice -n 15 cargo +1.95.0 build --manifest-path "$prefix/gst-plugins-rs/Cargo.toml" \
    --locked -p gst-plugin-webrtc -j 2
sh "$prefix/microduck/scripts/seed-policies.sh" "$prefix/policies"
for policy in velstand alpha_sitstand alpha_ground_pick ball_kick_left ball_kick_right roulade; do
    test -s "$prefix/policies/current/$policy.onnx"
done
# Quote paths through Python so prefixes containing spaces work in the generated launcher.
"$prefix/ito-venv/bin/python" - "$prefix" <<'PY'
import pathlib, shlex, sys
p = pathlib.Path(sys.argv[1])
q = lambda suffix: shlex.quote(str(p / suffix))
(p / 'microduck-sim').write_text(
    '#!/bin/sh\nset -eu\n'
    f'export GST_PLUGIN_PATH={q("gst-plugins-rs/target/debug")}${{GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}}\n'
    f'exec {q("ito-venv/bin/python")} -m drivers.microduck.sim '
    f'--microduck {q("microduck")} --rl {q("microduck_rl")} '
    f'--policies {q("policies/current")} "$@"\n'
)
(p / 'microduck-sim').chmod(0o755)
PY
printf '\nReady: %s/microduck-sim --viewer\n' "$prefix"
