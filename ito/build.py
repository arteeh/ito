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


RETRIEVAL_BASE = "https://download.europe.naverlabs.com/ComputerVision/MASt3R"
RETRIEVAL_NAME = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval"
# Upstream's load_retriever: the retrieval head and, beside it, the ASMK codebook it names.
RETRIEVAL_FILES = {
    f"{RETRIEVAL_NAME}_trainingfree.pth": (
        "000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50"
    ),
    f"{RETRIEVAL_NAME}_codebook.pkl": (
        "700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a"
    ),
}

# Ito's changes to upstream MASt3R-SLAM's Python, each with its reason (ATTRIBUTION.txt names
# them). Everything else runs as upstream ships it.
#
# FactorGraph (global_opt.py) keeps its edges and its solve's inputs in buffers allocated once,
# as SharedKeyframes keeps keyframes. Upstream concatenates them anew for every keyframe, so
# every solve asks Torch for blocks bigger than any it has cached: Torch's cache then grows
# without end, and on Windows, whose driver backs VRAM overflow with system memory instead of
# failing, into host RAM. Edges are stored two-way, each edge's forward and backward rows side
# by side, so a solve reads a view where upstream concatenates [all forward; all backward]: the
# same rows, so the same normal equations, summed in another order. Rays are constrained in
# place with constrain_points_to_ray's arithmetic.
FACTOR_GRAPH = (
    (
        "    constrain_points_to_ray,\n)",
        "    constrain_points_to_ray,\n    get_pixel_coords,\n)",
    ),
    (
        '    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):\n',
        "    def __init__(\n"
        '        self, model, frames: SharedKeyframes, K=None, device="cuda", max_edges=None\n'
        "    ):\n",
    ),
    (
        """\
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
""",
        """\
        # Ito: allocated once (ito/build.py FACTOR_GRAPH). run_backend adds at most 1 + k edges
        # per keyframe (the one before, k retrieved), relocalization at most k.
        most = (1 + config["retrieval"]["k"]) * frames.buffer
        rows = 2 * (most if max_edges is None else max_edges)
        n = frames.h * frames.w
        self.n_edges = 0
        self.ii2 = torch.zeros(rows, dtype=torch.long, device=self.device)
        self.jj2 = torch.zeros(rows, dtype=torch.long, device=self.device)
        self.idx2 = torch.zeros(rows, n, dtype=torch.long, device=self.device)
        self.valid2 = torch.zeros(rows, n, 1, dtype=torch.bool, device=self.device)
        self.Q2 = torch.zeros(rows, n, 1, dtype=torch.float32, device=self.device)
        self.Xs = torch.zeros(frames.buffer, n, 3, dtype=torch.float32, device=self.device)
        self.Cs = torch.zeros(frames.buffer, n, 1, dtype=torch.float32, device=self.device)
        self.rays = None
""",
    ),
    (
        """\
        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])
""",
        """\
        start = 2 * self.n_edges
        end = start + 2 * len(ii_tensor)
        if end > len(self.ii2):
            raise RuntimeError("FactorGraph is full")
        for rows, forward, backward in (
            (self.ii2, ii_tensor, jj_tensor),
            (self.jj2, jj_tensor, ii_tensor),
            (self.idx2, idx_i2j, idx_j2i),
            (self.valid2, valid_match_j, valid_match_i),
            (self.Q2, Qj, Qi),
        ):
            rows[start:end:2] = forward
            rows[start + 1 : end : 2] = backward
        self.n_edges += len(ii_tensor)
""",
    ),
    (
        """\
    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)
""",
        """\
    @property
    def ii(self):
        return self.ii2[: 2 * self.n_edges : 2]

    @property
    def jj(self):
        return self.jj2[: 2 * self.n_edges : 2]

    def get_unique_kf_idx(self):
        return torch.unique(self.ii2[: 2 * self.n_edges], sorted=True)
""",
    ),
    (
        """\
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj
""",
        """\
        rows = 2 * self.n_edges
        edges = self.ii2, self.jj2, self.idx2, self.valid2, self.Q2
        return tuple(edge[:rows] for edge in edges)
""",
    ),
    (
        """\
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])
""",
        """\
        n = len(unique_kf_idx)
        with self.frames.lock:
            Xs = torch.index_select(self.frames.X, 0, unique_kf_idx, out=self.Xs[:n])
            T_WCs = lietorch.Sim3(self.frames.T_WC[unique_kf_idx])
            Cs = torch.index_select(self.frames.C, 0, unique_kf_idx, out=self.Cs[:n])
            Cs /= self.frames.N[unique_kf_idx].view(-1, 1, 1)
""",
    ),
    (
        """\
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)
""",
        """\
        img_size = self.frames[0].img.shape[-2:]
        if self.rays is None:
            uv = get_pixel_coords(1, img_size, device=Xs.device, dtype=Xs.dtype).view(-1, 2)
            self.rays = ((uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1])
        for axis, ray in enumerate(self.rays):
            Xs[..., axis].copy_(Xs[..., 2]).mul_(ray)
""",
    ),
)


# FrameTracker takes any error in its pose solve for a lost frame, CUDA out of memory too; it
# then relocalises, which needs more memory still, and nobody is told. Ito re-raises that one, so
# it reaches the worker's out-of-VRAM handling and its message.
TRACKER = (
    (
        '        except Exception as e:\n            print(f"Cholesky failed {frame.frame_id}")\n',
        "        except torch.cuda.OutOfMemoryError:\n"
        "            raise\n"
        "        except Exception as e:\n"
        '            print(f"Cholesky failed {frame.frame_id}")\n',
    ),
)


def adapt(path, replacements):
    code = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if code.count(old) != 1:
            raise RuntimeError(f"MASt3R-SLAM changed upstream; cannot adapt {path}")
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
    native["asmk.hamming"] = compile_hamming(slam, cache, work)
    return native, str(torch.__version__)


def compile_hamming(slam, cache, work):
    """asmk's `hamming` module (ASMK, the retrieval behind MASt3R-SLAM's loop closure and
    relocalisation), from the C its authors generated (cython/hamming.c), built as asmk's own
    setup.py builds it. Loaded as asmk.hamming by mast3r_runtime.prepare."""
    from setuptools import Distribution, Extension

    report("Building asmk's hamming module", -1)
    output = work / "native" / "asmk"
    if output.exists():
        shutil.rmtree(output)
    source = slam / "thirdparty/mast3r/asmk/cython/hamming.c"
    try:
        command = Distribution(
            dict(name="asmk", ext_modules=[Extension("asmk.hamming", [str(source)])])
        ).get_command_obj("build_ext")
        command.build_lib = str(output / "lib")
        command.build_temp = str(output / "temp")
        command.ensure_finalized()
        command.run()
        (built,) = (output / "lib" / "asmk").glob("hamming*")
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise RuntimeError("Could not build asmk's hamming module; check the C compiler") from exc
    target = cache / "native" / built.name
    shutil.copy2(built, target)
    return target.relative_to(cache).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODELS, help="release models directory")
    args = parser.parse_args()
    cache = args.output.resolve()
    work = Path(__file__).resolve().parents[1] / ".slam-build"
    cache.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
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
    adapt(slam / "mast3r_slam/global_opt.py", FACTOR_GRAPH)
    adapt(slam / "mast3r_slam/tracker.py", TRACKER)
    # Windows' `long` is 32 bits; the kernels index with it.
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
    for name, digest in RETRIEVAL_FILES.items():
        download(
            f"{RETRIEVAL_BASE}/{name}",
            cache / "retrieval" / name,
            report,
            f"MASt3R retrieval {name.rsplit('_', 1)[-1]} (CC BY-NC-SA; non-commercial)",
            digest,
        )
    for name in ("LICENSE", "NOTICE", "CHECKPOINTS_NOTICE"):
        shutil.copy2(slam / "thirdparty/mast3r" / name, weights / name)
    (weights / "ATTRIBUTION.txt").write_text(
        f"MASt3R, Copyright 2024-present NAVER Corp.\n{MODEL}\nRevision: {MODEL_REV}\n"
        f"https://huggingface.co/{MODEL}/tree/{MODEL_REV}\n"
        "Unmodified safetensors weights. CC BY-NC-SA 4.0; see LICENSE, NOTICE and "
        "CHECKPOINTS_NOTICE for attribution and training-dataset restrictions.\n"
        f"retrieval/: MASt3R's retrieval head and ASMK codebook, {RETRIEVAL_BASE}/"
        f"{RETRIEVAL_NAME}_*. Unmodified. CC BY-NC-SA 4.0, as above.\n"
        f"slam/: MASt3R-SLAM revision {SLAM_REV}. Ito source adaptations: FactorGraph keeps "
        "its edges and solve inputs in buffers allocated once (global_opt.py), the frame "
        "tracker re-raises CUDA out-of-memory errors (tracker.py), fixed-width CUDA indices, "
        "distinct lietorch CPU kernel symbols, current-stream CUDA launches. Ito feeds it frames "
        "of upstream's 512-pixel long side, or, at an Input resolution below 100 %, 448, 368 or "
        "256 (not upstream's).\n",
        encoding="utf-8",
    )
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
