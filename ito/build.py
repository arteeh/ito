import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from ito import clock
from ito.reconstruction.mast3r_runtime import BUNDLE_VERSION, MODEL_FILES, MODELS, abi

BUILD_COMMAND = "uv run --python 3.12 --extra slam python -m ito.build"

SLAM_REV = "e6f4e3d474fad0e11f561482012be864ba8c3f17"
LIE_REV = "e7df86554156b36846008d8ddbcc4d8521a16554"
EIGEN_REV = "bddaa99e151244402c6e804e01b970288650da6b"
MODEL_REV = "06e7259f34c3060f322df5cb0c7b9094f57e41fc"
MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEPTH_REV = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DEPTH_MODEL = "depth-anything/DA3-BASE"
DEPTH_MODEL_REV = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"
DEPTH_FILES = {
    "model.safetensors": "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
    "config.json": "5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5",
}
MOGE_REV = "74fbce054ebed49800de42d0ad0e83495065719a"
UTILS3D_REV = "62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
MOGE_MODEL = "Ruicheng/moge-2-vitb-normal"
MOGE_MODEL_REV = "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"
MOGE_FILES = {"model.pt": "16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6"}


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
                now = clock.now()
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


def source(cache, work, name, url, digest, report):
    root = cache / name
    archive = download(url, work / f"{name}.tar.gz", report, name, digest)
    if root.exists():
        shutil.rmtree(root)
    report(f"Preparing {name}", -1)
    # Not tempfile: on Windows its owner-only ACL is inherited by everything extracted,
    # so a release built from an elevated shell would be unreadable by the pilot.
    staging = cache / f".{name}-extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
        (extracted,) = staging.iterdir()
        extracted.rename(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


# Robot-first tracking (ito/reconstruction/mast3r_frontend.py): FrameTracker.solve adds the
# fused prediction as a prior to its Gauss-Newton normal equations, so a frame with few
# matches yields a pose leaning on the robot instead of being dropped as lost. Without a
# prior (the vision-only tracker) upstream's behaviour is unchanged.
RELATIVE_SETUP = (
    "        # Solving for relative pose without scale!\n        T_CkCf = T_WCk.inv() * T_WCf\n"
)
TRACKER_ADAPTATIONS = (
    (
        "        self.device = device\n\n        self.reset_idx_f2k()\n",
        "        self.device = device\n\n        self.reset_idx_f2k()\n"
        "        # Ito: (predicted T_WCf Sim3, 7 information weights) or None.\n"
        "        self.prior = None\n"
        "        self.prior_relative = None\n"
        "        self.match_frac = 0.0\n",
    ),
    (
        "        match_frac = valid_opt.sum() / valid_opt.numel()\n"
        '        if match_frac < self.cfg["min_match_frac"]:\n',
        "        match_frac = valid_opt.sum() / valid_opt.numel()\n"
        "        self.match_frac = float(match_frac)\n"
        '        if match_frac < self.cfg["min_match_frac"] and self.prior is None:\n',
    ),
    (
        "        self.keyframes[len(self.keyframes) - 1] = keyframe\n",
        '        if getattr(self.keyframes, "reference", None) is None:\n'
        "            self.keyframes[len(self.keyframes) - 1] = keyframe\n",
    ),
    (
        "    def solve(self, sqrt_info, r, J):\n",
        "    def solve(self, sqrt_info, r, J, T_CkCf=None):\n",
    ),
    (
        "        cost = 0.5 * (b.T @ b).item()\n",
        "        cost = 0.5 * (b.T @ b).item()\n"
        "        if self.prior_relative is not None and T_CkCf is not None:\n"
        "            # retr is Exp(tau) * X: the prior's residual moves one for one with tau.\n"
        "            weight = self.prior[1].to(H.dtype)\n"
        "            r_p = (T_CkCf * self.prior_relative.inv()).log().view(-1, 1).to(H.dtype)\n"
        "            H = H + torch.diag(weight)\n"
        "            g = g - weight.view(-1, 1) * r_p\n"
        "            cost += 0.5 * (weight.view(-1, 1) * r_p * r_p).sum().item()\n",
    ),
    (
        RELATIVE_SETUP,
        RELATIVE_SETUP + "        self.prior_relative = (\n"
        "            None if self.prior is None else T_WCk.inv() * self.prior[0]\n"
        "        )\n",
    ),
    (
        "            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)\n",
        "            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J, T_CkCf)\n",
    ),
    (
        "            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)\n",
        "            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J, T_CkCf)\n",
    ),
)


def adapt_tracker(path):
    code = path.read_text(encoding="utf-8")
    for old, new in TRACKER_ADAPTATIONS:
        expected = 2 if old == RELATIVE_SETUP else 1
        if code.count(old) != expected:
            raise RuntimeError(f"MASt3R-SLAM tracker changed upstream; cannot adapt {path}")
        code = code.replace(old, new)
    path.write_text(code, encoding="utf-8")


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

    # Upstream launches on stream 0. Every allocation and launch must follow
    # Torch's current stream for Ito's nonblocking reconstruction stream to be safe.
    kernels = [
        lie / "lietorch/src/lietorch_gpu.cu",
        *(slam / "mast3r_slam/backend/src").glob("*.cu"),
        slam / "thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu",
    ]
    for path in kernels:
        code = path.read_text(encoding="utf-8")
        launches = re.findall(r"<<<([^<>]+)>>>", code)
        if not launches:
            raise RuntimeError(f"No CUDA launches to adapt in {path}")

        def current_stream(match, *, path=path):
            launch = match[1]
            arguments = len(launch.split(","))
            if arguments not in (2, 3):
                raise RuntimeError(f"Unexpected CUDA launch in {path}: {launch}")
            shared = ", 0" if arguments == 2 else ""
            return f"<<<{launch}{shared}, at::cuda::getCurrentCUDAStream()>>>"

        code = re.sub(r"<<<([^<>]+)>>>", current_stream, code)
        path.write_text("#include <ATen/cuda/CUDAContext.h>\n" + code, encoding="utf-8")

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


def depth_bundle(cache, work):
    """models/depth (ito/reconstruction/depth.py): Depth Anything 3's model code and the
    DA3-BASE weights, both Apache-2.0."""
    root = source(
        cache,
        work,
        "depth-source",
        f"https://codeload.github.com/ByteDance-Seed/Depth-Anything-3/tar.gz/{DEPTH_REV}",
        "98aa2dd53ab44b96cef5190ae4841c6ae51797d099b784e08e12a1a55bd3a69b",
        report,
    )
    depth = cache / "depth"
    if depth.exists():
        shutil.rmtree(depth)
    shutil.copytree(
        root / "src/depth_anything_3",
        depth / "depth_anything_3",
        ignore=shutil.ignore_patterns("app", "services", "bench", "__pycache__"),
    )
    shutil.copy2(root / "LICENSE", depth / "LICENSE")
    shutil.rmtree(root)
    for name, digest in DEPTH_FILES.items():
        download(
            f"https://huggingface.co/{DEPTH_MODEL}/resolve/{DEPTH_MODEL_REV}/{name}",
            depth / name,
            report,
            f"DA3-BASE {name} (Apache-2.0)",
            digest,
        )
    (depth / "ATTRIBUTION.txt").write_text(
        f"Depth Anything 3, ByteDance Seed. Code revision {DEPTH_REV}, "
        f"https://github.com/ByteDance-Seed/Depth-Anything-3\n"
        f"{DEPTH_MODEL} revision {DEPTH_MODEL_REV}, "
        f"https://huggingface.co/{DEPTH_MODEL}/tree/{DEPTH_MODEL_REV}\n"
        "Unmodified source and safetensors weights, Apache-2.0; see LICENSE.\n",
        encoding="utf-8",
    )


def moge_bundle(cache, work):
    """models/moge (ito/reconstruction/depth.py, MonoGeometry): MoGe-2's model code and
    utils3d-moge (both MIT) and the MoGe-2 ViT-B weights (MIT)."""
    root = source(
        cache,
        work,
        "moge-source",
        f"https://codeload.github.com/microsoft/MoGe/tar.gz/{MOGE_REV}",
        "cc6f3525cec7694b97d6d9af7d8854f546cda082f3abc944be17dfb6ad892309",
        report,
    )
    utils = source(
        cache,
        work,
        "utils3d-source",
        f"https://codeload.github.com/EasternJournalist/utils3d-moge/tar.gz/{UTILS3D_REV}",
        "a6917f2e641b1221daf2e2df06f4605e1754defa07abf4059fc514ddbebffc68",
        report,
    )
    moge = cache / "moge"
    if moge.exists():
        shutil.rmtree(moge)
    shutil.copytree(
        root / "moge",
        moge / "moge",
        ignore=shutil.ignore_patterns(
            "scripts", "train", "test", "gradio_3d_viewer", "__pycache__"
        ),
    )
    shutil.copytree(
        utils / "utils3d_moge", moge / "utils3d_moge", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(root / "LICENSE", moge / "LICENSE")
    shutil.copy2(utils / "LICENSE", moge / "LICENSE-utils3d")
    shutil.rmtree(root)
    shutil.rmtree(utils)
    for name, digest in MOGE_FILES.items():
        download(
            f"https://huggingface.co/{MOGE_MODEL}/resolve/{MOGE_MODEL_REV}/{name}",
            moge / name,
            report,
            f"MoGe-2 ViT-B {name} (MIT)",
            digest,
        )
    (moge / "ATTRIBUTION.txt").write_text(
        f"MoGe, Microsoft. Code revision {MOGE_REV}, https://github.com/microsoft/MoGe (MIT;\n"
        "its DINOv2 code, moge/model/dinov2, Apache-2.0).\n"
        f"utils3d-moge revision {UTILS3D_REV}, https://github.com/EasternJournalist/utils3d-moge"
        " (MIT).\n"
        f"{MOGE_MODEL} revision {MOGE_MODEL_REV}, "
        f"https://huggingface.co/{MOGE_MODEL}/tree/{MOGE_MODEL_REV} (MIT).\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODELS, help="release models directory")
    parser.add_argument(
        "--depth-only", action="store_true", help="only (re)fetch the depth model into it"
    )
    parser.add_argument(
        "--moge-only", action="store_true", help="only (re)fetch the MoGe-2 model into it"
    )
    args = parser.parse_args()
    cache = args.output.resolve()
    work = Path(__file__).resolve().parents[1] / ".slam-build"
    cache.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if args.depth_only or args.moge_only:
        if args.depth_only:
            depth_bundle(cache, work)
        if args.moge_only:
            moge_bundle(cache, work)
        return
    manifest = cache / "manifest.json"
    manifest.unlink(missing_ok=True)
    slam = source(
        cache,
        work,
        "slam",
        f"https://codeload.github.com/rmurai0610/MASt3R-SLAM/tar.gz/{SLAM_REV}",
        "787099d1b8eeba2a4e6639746e7f561ac77bff9c7c9f8faf722c488e04a316f1",
        report,
    )
    source(
        cache,
        work,
        "lie",
        f"https://codeload.github.com/princeton-vl/lietorch/tar.gz/{LIE_REV}",
        "856ff3792a4d4b5343ad642fd699a077a556758526d9b231616fc277b170c847",
        report,
    )
    source(
        cache,
        work,
        "eigen",
        f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_REV}/eigen-{EIGEN_REV}.tar.gz",
        "7a246279efcaf15464aac710ca36fde772000fa0a16082a8c03dfe78c15dce93",
        report,
    )
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
    adapt_tracker(slam / "mast3r_slam/tracker.py")
    for path in (slam / "mast3r_slam/backend/src").glob("*.cu"):
        original = path.read_text(encoding="utf-8")
        fixed = re.sub(r"\blong\b", "int64_t", original)
        if fixed != original:
            path.write_text(fixed, encoding="utf-8")
    # MSVC gives CPU templates and NVCC's host launch stubs identical symbols.
    # The linker can select a CPU implementation that then dereferences CUDA memory.
    cpu = cache / "lie/lietorch/src/lietorch_cpu.cpp"
    cpu.write_text(
        re.sub(r"\b(\w+)_kernel\b", r"\1_cpu_kernel", cpu.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    weights = cache
    base = f"https://huggingface.co/{MODEL}/resolve/{MODEL_REV}"
    download(
        f"{base}/config.json",
        weights / "config.json",
        report,
        "model config",
        MODEL_FILES["config.json"],
    )
    download(
        f"{base}/model.safetensors",
        weights / "model.safetensors",
        report,
        "MASt3R weights (CC BY-NC-SA; non-commercial)",
        MODEL_FILES["model.safetensors"],
    )
    for name in ("LICENSE", "NOTICE", "CHECKPOINTS_NOTICE"):
        shutil.copy2(slam / "thirdparty/mast3r" / name, weights / name)
    (weights / "ATTRIBUTION.txt").write_text(
        f"MASt3R, Copyright 2024-present NAVER Corp.\n{MODEL}\nRevision: {MODEL_REV}\n"
        f"https://huggingface.co/{MODEL}/tree/{MODEL_REV}\n"
        "Unmodified safetensors weights. CC BY-NC-SA 4.0; see LICENSE, NOTICE and "
        "CHECKPOINTS_NOTICE for attribution and training-dataset restrictions.\n"
        "Ito source adaptations: lazy retrieval import, fixed-width CUDA indices, "
        "distinct lietorch CPU kernel symbols, current-stream CUDA launches, "
        "and a pose prior in the frame tracker.\n",
        encoding="utf-8",
    )
    depth_bundle(cache, work)
    moge_bundle(cache, work)
    if (cache / "native").exists():
        shutil.rmtree(cache / "native")
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
