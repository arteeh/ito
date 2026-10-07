"""Fetch, verify and compile the offline SLAM bundle before packaging Ito."""

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from ito.reconstruction.mast3r_runtime import BUILD_COMMAND, BUNDLE, BUNDLE_VERSION, abi

SLAM_REV = "e6f4e3d474fad0e11f561482012be864ba8c3f17"
LIE_REV = "e7df86554156b36846008d8ddbcc4d8521a16554"
EIGEN_REV = "bddaa99e151244402c6e804e01b970288650da6b"
MODEL_REV = "06e7259f34c3060f322df5cb0c7b9094f57e41fc"
MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def report(message, progress=-1):
    print(message, flush=True)


def download(url, path, report, label, sha256=None):
    if path.is_file():
        with path.open("rb") as existing:
            valid = not sha256 or hashlib.file_digest(existing, "sha256").hexdigest() == sha256
        if valid:
            report(f"Verified {label}")
            return path
        path.unlink()
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
        report(f"Verified {label}")
    finally:
        partial.unlink(missing_ok=True)
    return path


def source(cache, name, url, digest, report):
    root = cache / name
    archive = download(url, cache.parent / ".slam-build" / f"{name}.tar.gz", report, name, digest)
    if root.exists():
        shutil.rmtree(root)
    report(f"Preparing {name}", -1)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        with tarfile.open(archive) as tar:
            tar.extractall(temporary, filter="data")
        (extracted,) = Path(temporary).iterdir()
        extracted.rename(root)
    return root


def compile_extensions(cache, work):
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
    except ImportError:
        report("CUDA compilation skipped: install build dependencies with " + BUILD_COMMAND)
        return {}, None
    nvcc = Path(CUDA_HOME or "") / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    if CUDA_HOME is None or not nvcc.is_file():
        report(
            "CUDA compilation skipped: CUDA Toolkit 12.4 (nvcc) is unavailable. "
            "Sources and weights are verified; SLAM will use the flat camera-feed fallback."
        )
        return {}, None
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("CUDA compilation requires Python 3.12: " + BUILD_COMMAND)
    slam, lie, eigen = (cache / name for name in ("slam", "lie", "eigen"))
    cxx = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3"]
    os.environ.setdefault("MAX_JOBS", "2")

    # A release must run on other NVIDIA GPUs and build on GPU-less CI with nvcc.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0+PTX")
    native = {}

    def build(name, files, includes=()):
        report(f"Building {name} CUDA kernels", -1)
        directory = work / "native" / name
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
                f"Could not build {name}; check CUDA 12.4 and the C++ compiler. See build log"
            ) from exc
        target = cache / "native" / Path(module.__file__).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module.__file__, target)
        native[name] = target.relative_to(cache).as_posix()

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
    return native, str(torch.__version__)


def main():
    cache = BUNDLE
    work = cache.parent / ".slam-build"
    cache.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must never appear ready to the pilot or packager.
    manifest = cache / "manifest.json"
    manifest.unlink(missing_ok=True)
    slam = source(
        cache,
        "slam",
        f"https://codeload.github.com/rmurai0610/MASt3R-SLAM/tar.gz/{SLAM_REV}",
        "787099d1b8eeba2a4e6639746e7f561ac77bff9c7c9f8faf722c488e04a316f1",
        report,
    )
    source(
        cache,
        "lie",
        f"https://codeload.github.com/princeton-vl/lietorch/tar.gz/{LIE_REV}",
        "856ff3792a4d4b5343ad642fd699a077a556758526d9b231616fc277b170c847",
        report,
    )
    source(
        cache,
        "eigen",
        f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_REV}/eigen-{EIGEN_REV}.tar.gz",
        "7a246279efcaf15464aac710ca36fde772000fa0a16082a8c03dfe78c15dce93",
        report,
    )
    # Retain all upstream licences. Retrieval's ASMK database is unnecessary for a
    # bounded live map: recovery searches the retained keyframes directly.
    utils = slam / "mast3r_slam/mast3r_utils.py"
    code = utils.read_text(encoding="utf-8")
    eager = "from mast3r_slam.retrieval_database import RetrievalDatabase\n"
    if eager.rstrip() in code.splitlines():
        code = code.replace(eager, "").replace(
            "    retriever_path = (",
            "    from mast3r_slam.retrieval_database import RetrievalDatabase\n"
            "    retriever_path = (",
        )
        utils.write_text(code, encoding="utf-8")
    for path in (slam / "mast3r_slam/backend/src").glob("*.cu"):
        original = path.read_text(encoding="utf-8")
        fixed = re.sub(r"\blong\b", "int64_t", original)
        if fixed != original:
            path.write_text(fixed, encoding="utf-8")
    weights = cache / "weights"
    base = f"https://huggingface.co/{MODEL}/resolve/{MODEL_REV}"
    download(
        f"{base}/config.json",
        weights / "config.json",
        report,
        "model config",
        "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4",
    )
    download(
        f"{base}/model.safetensors",
        weights / "model.safetensors",
        report,
        "MASt3R weights (CC BY-NC-SA; non-commercial)",
        "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb",
    )
    for name in ("LICENSE", "NOTICE", "CHECKPOINTS_NOTICE"):
        shutil.copy2(slam / "thirdparty/mast3r" / name, weights / name)
    (weights / "ATTRIBUTION.txt").write_text(
        f"MASt3R, Copyright 2024-present NAVER Corp.\n{MODEL}\nRevision: {MODEL_REV}\n"
        f"https://huggingface.co/{MODEL}/tree/{MODEL_REV}\n"
        "Unmodified safetensors weights. CC BY-NC-SA 4.0; see LICENSE, NOTICE and "
        "CHECKPOINTS_NOTICE for attribution and training-dataset restrictions.\n"
        "Ito source adaptations: lazy retrieval import and fixed-width CUDA indices.\n",
        encoding="utf-8",
    )
    native, torch_version = compile_extensions(cache, work)
    files = {
        p.relative_to(cache).as_posix(): p.stat().st_size
        for p in cache.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    manifest.write_text(
        json.dumps(
            dict(
                version=BUNDLE_VERSION,
                abi=abi(),
                torch=torch_version,
                native=native,
                revisions=dict(slam=SLAM_REV, lie=LIE_REV, eigen=EIGEN_REV, model=MODEL_REV),
                files=files,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report(f"SLAM bundle {'ready' if native else 'incomplete (CUDA compilation skipped)'}: {cache}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        sys.exit(f"SLAM build failed: {exc}")
