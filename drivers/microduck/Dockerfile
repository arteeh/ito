# The virtual Microduck's robot side (compose.yaml): Pollen's daemons at the third_party/ pins and
# Ito's Microduck driver. Build context: the repository root.
FROM ubuntu:26.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential pkg-config curl ca-certificates git libssl-dev libudev-dev libclang-dev clang \
    libasound2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.95.0
ENV PATH=/root/.cargo/bin:$PATH
COPY third_party/microduck /src/microduck
COPY third_party/gst-plugins-rs /src/gst-plugins-rs
RUN --mount=type=cache,target=/root/.cargo/registry --mount=type=cache,target=/src/target \
    cargo build --release --locked --target-dir /src/target \
        --manifest-path /src/microduck/Cargo.toml -p robotd -p robotctl -p mediad -p tof \
    && cargo build --release --locked --target-dir /src/target \
        --manifest-path /src/gst-plugins-rs/Cargo.toml -p gst-plugin-webrtc \
    && mkdir -p /opt/microduck/bin /opt/microduck/gst \
    && cp /src/target/release/robotd /src/target/release/robotctl /src/target/release/mediad \
        /src/target/release/tofd /opt/microduck/bin/ \
    && cp /src/target/release/libgstrswebrtc.so /opt/microduck/gst/
RUN sh /src/microduck/scripts/seed-policies.sh /opt/microduck/policies \
    && test -s /opt/microduck/policies/current/velstand.onnx

FROM ubuntu:26.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates libssl3t64 libudev1 libasound2t64 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-nice \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
COPY --from=build /opt/microduck /opt/microduck
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python \
    GST_PLUGIN_PATH=/opt/microduck/gst PYTHONPATH=/opt/ito PATH=/opt/venv/bin:$PATH
COPY drivers/microduck/pyproject.toml drivers/microduck/uv.lock /opt/ito/drivers/microduck/
RUN uv sync --frozen --no-install-project --project /opt/ito/drivers/microduck --python 3.12 \
    && uv pip install --python /opt/venv onnxruntime==1.24.4
CMD ["python", "-m", "drivers.microduck.robot"]
