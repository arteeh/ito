"""Pinned upstream sources and native extensions, isolated in the pilot's cache.

Upstream's installer pulls in its GUI and assumes GCC/LP64. Build its actual kernels
with PyTorch instead, using MSVC flags and fixed-width indices on Windows. All of
this runs in the reconstruction child; the display continues showing live video.
"""

import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

SLAM_REV = "e6f4e3d474fad0e11f561482012be864ba8c3f17"
LIE_REV = "e7df86554156b36846008d8ddbcc4d8521a16554"
EIGEN_REV = "bddaa99e151244402c6e804e01b970288650da6b"
MODEL_REV = "06e7259f34c3060f322df5cb0c7b9094f57e41fc"
MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def download(url, path, report, label, sha256=None):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    # Restart an interrupted download; only a complete, verified file gets its final name.
    report(f"Downloading {label}", 0)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            received, notified = 0, 0.0
            while block := response.read(1024 * 1024):
                out.write(block)
                digest.update(block)
                received += len(block)
                now = time.monotonic()
                if now - notified > 0.2:
                    report(
                        f"Downloading {label}: {received / 1e6:.0f} MB"
                        + (f" / {total / 1e6:.0f} MB" if total else ""),
                        received / total if total else -1,
                    )
                    notified = now
            if total and received != total:
                raise OSError(f"Incomplete {label} download")
        if sha256 and digest.hexdigest() != sha256:
            raise OSError(f"Checksum mismatch for {label}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def source(cache, name, url, digest, report):
    root = cache / name
    if root.is_dir():
        return root
    archive = download(url, cache / f"{name}.tar.gz", report, name, digest)
    report(f"Preparing {name}", -1)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        with tarfile.open(archive) as tar:
            tar.extractall(temporary, filter="data")
        (extracted,) = Path(temporary).iterdir()
        extracted.rename(root)
    return root


def prepare(report):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "MASt3R-SLAM CUDA needs Python 3.12; use uv run --python 3.12 --extra slam"
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MASt3R-SLAM needs NVIDIA CUDA; install with uv sync --extra slam"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("MASt3R-SLAM needs an NVIDIA CUDA device and driver")

    from filelock import FileLock
    from platformdirs import user_cache_path
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("MASt3R-SLAM needs CUDA Toolkit 12.4 (nvcc) to build its kernels")
    # Version this directory when patches change; never mutate a loaded source tree.
    cache = user_cache_path("ito") / "mast3r-slam" / (SLAM_REV[:12] + "-1")
    cache.mkdir(parents=True, exist_ok=True)
    report("Preparing MASt3R-SLAM (first use builds CUDA kernels)", -1)
    with FileLock(str(cache / "prepare.lock"), timeout=600):
        slam = source(
            cache,
            "slam",
            f"https://codeload.github.com/rmurai0610/MASt3R-SLAM/tar.gz/{SLAM_REV}",
            "787099d1b8eeba2a4e6639746e7f561ac77bff9c7c9f8faf722c488e04a316f1",
            report,
        )
        lie = source(
            cache,
            "lie",
            f"https://codeload.github.com/princeton-vl/lietorch/tar.gz/{LIE_REV}",
            "856ff3792a4d4b5343ad642fd699a077a556758526d9b231616fc277b170c847",
            report,
        )
        eigen = source(
            cache,
            "eigen",
            f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_REV}/eigen-{EIGEN_REV}.tar.gz",
            "7a246279efcaf15464aac710ca36fde772000fa0a16082a8c03dfe78c15dce93",
            report,
        )
        # Retain all upstream licences. Retrieval's ASMK database is unnecessary for a
        # bounded live map: recovery searches the retained keyframes directly.
        utils = slam / "mast3r_slam/mast3r_utils.py"
        code = utils.read_text()
        eager = "from mast3r_slam.retrieval_database import RetrievalDatabase\n"
        if eager.rstrip() in code.splitlines():
            code = code.replace(eager, "").replace(
                "    retriever_path = (",
                "    from mast3r_slam.retrieval_database import RetrievalDatabase\n"
                "    retriever_path = (",
            )
            utils.write_text(code)
        for path in (slam / "mast3r_slam/backend/src").glob("*.cu"):
            original = path.read_text()
            fixed = re.sub(r"\blong\b", "int64_t", original)
            if fixed != original:
                path.write_text(fixed)
        cxx = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3"]
        os.environ.setdefault("MAX_JOBS", "2")

        # Include the ABI and GPU in our build directory. The outer OS lock also
        # lets an interrupted first build safely discard PyTorch's stale baton.
        gpu = torch.cuda.get_device_capability()
        tag = f"{sys.implementation.cache_tag}-{torch.__version__}-{gpu}"
        tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", tag)

        def build(name, files, includes=()):
            report(f"Building/loading {name} CUDA kernels; first build can take minutes", -1)
            directory = cache / "native" / tag / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "lock").unlink(missing_ok=True)
            try:
                module = load(
                    name,
                    list(map(str, files)),
                    extra_include_paths=list(map(str, includes)),
                    build_directory=str(directory),
                    extra_cflags=cxx,
                    extra_cuda_cflags=["-O3"],
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Could not build {name}; check CUDA 12.4 and the C++ compiler. See pilot log"
                ) from exc
            sys.modules[name] = module
            return module

        backend = slam / "mast3r_slam/backend"
        build(
            "lietorch_backends",
            [
                lie / "lietorch/src" / f
                for f in (
                    "lietorch.cpp",
                    "lietorch_cpu.cpp",
                    "lietorch_gpu.cu",
                )
            ],
            [lie / "lietorch/include", eigen],
        )
        build(
            "mast3r_slam_backends",
            [
                backend / "src" / f
                for f in (
                    "gn.cpp",
                    "gn_kernels.cu",
                    "matching_kernels.cu",
                )
            ],
            [backend / "include", eigen],
        )
        croco = slam / "thirdparty/mast3r/dust3r/croco/models/curope"
        build("curope", [croco / "curope.cpp", croco / "kernels.cu"])
    sys.path[:0] = [str(slam), str(lie), str(slam / "thirdparty/mast3r")]
    return slam, cache


def load_model(cache, report):
    from filelock import FileLock
    from mast3r.model import AsymmetricMASt3R
    from safetensors.torch import load_file

    weights = cache.parent / MODEL_REV
    weights.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{MODEL}/resolve/{MODEL_REV}"
    with FileLock(str(weights / "download.lock"), timeout=1800):
        config = download(f"{base}/config.json", weights / "config.json", report, "model config")
        path = download(
            f"{base}/model.safetensors",
            weights / "model.safetensors",
            report,
            "MASt3R weights (CC BY-NC-SA; non-commercial)",
            "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb",
        )
    report("Loading MASt3R weights onto CUDA", -1)
    # Safetensors avoids executing a pickle checkpoint, and pins the exact published model.
    model = AsymmetricMASt3R(**json.loads(config.read_text()))
    model.load_state_dict(load_file(path), strict=True)
    return model.eval().to("cuda")
