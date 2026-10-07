"""Build dist/ito-windows-x64.zip: unzip anywhere, run ito.exe. Developer-only.

From a VS 2022 x64 developer shell with CUDA 12.4 (the toolchain ito.build needs):

    uv run --python 3.12 --extra slam python release/windows.py [--models DIR]

Packaging is a relocatable python-build-standalone runtime (the one uv manages) with the
locked dependencies installed into it, started by a small native ito.exe. Freezers such as
PyInstaller have to rediscover what torch, CuPy, imgui-bundle, aiortc, MuJoCo and pyopenxr
load at runtime, and MASt3R-SLAM imports its own sources from models\\ at run time; a
plain interpreter with ordinary site-packages runs exactly what was tested from source.

    ito\\ito.exe       release/ito.c: runs runtime\\pythonw.exe -I -m ito.app
    ito\\runtime\\      CPython 3.12 + site-packages
    ito\\models\\       MASt3R weights, licences, sources and CUDA kernels (ito.build)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from ito.reconstruction.mast3r_runtime import BUNDLE_VERSION, abi

ROOT = Path(__file__).resolve().parents[1]
NAME = "ito-windows-x64"


def run(*command, **options):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **options)


def runtime(stage, work):
    """Copy the uv-managed interpreter this script runs on, then install the lockfile into it."""
    managed = Path(subprocess.check_output(["uv", "python", "dir"], text=True).strip())
    base = Path(sys.base_prefix)
    if base.parent.resolve() != managed.resolve() or sys.version_info[:2] != (3, 12):
        raise SystemExit(f"Run with uv's managed Python 3.12 (found {base}): {__doc__}")
    target = stage / "runtime"
    shutil.copytree(base, target, ignore=shutil.ignore_patterns("__pycache__"))
    site = target / "Lib" / "site-packages"
    for path in site.iterdir():
        if path.name != "README.txt":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    lock = work / "pylock.toml"
    run(
        "uv", "export", "--frozen", "--no-dev", "--extra", "slam", "--no-emit-project",
        "--format", "pylock.toml", "--output-file", lock,
        cwd=ROOT,
    )  # fmt: skip
    run("uv", "build", "--wheel", "--out-dir", work, cwd=ROOT)
    (wheel,) = work.glob("ito-*.whl")
    python = target / "python.exe"
    # --target writes into this copy only, never into the interpreter it was copied from.
    for requirement in (["-r", lock], ["--no-deps", wheel]):
        run(
            "uv", "pip", "install", "--python", python, "--target", site,
            "--link-mode", "copy", "--compile-bytecode", *requirement,
        )  # fmt: skip
    # Console scripts with absolute paths; libraries and headers for compiling against
    # torch (the kernels in models\ are prebuilt); Android OpenXR loaders and layers.
    unused = [site / "bin", site / "torch/include", *site.rglob("*.lib")]
    for path in unused + list(site.glob("xr/**/android*")):
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)


def launcher(stage, work):
    if not shutil.which("cl"):
        raise SystemExit("cl.exe not found: run from a VS 2022 x64 developer shell")
    run(
        "cl", "/nologo", "/O2", "/W4", "/WX", "/DUNICODE", ROOT / "release/ito.c",
        f"/Fe{stage / 'ito.exe'}", f"/Fo{work / 'ito.obj'}",
        "/link", "/SUBSYSTEM:WINDOWS", "user32.lib",
        cwd=work,
    )  # fmt: skip


def models(stage, source):
    target = stage / "models"
    if source is None:
        run(sys.executable, "-m", "ito.build", "--output", target, cwd=ROOT)
    else:
        print(f"Copying models from {source}", flush=True)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != BUNDLE_VERSION or manifest["abi"] != abi() or not manifest["native"]:
        raise SystemExit(f"{target} is not a complete bundle {BUNDLE_VERSION} for {abi()}")
    for name, size in manifest["files"].items():
        if (target / name).stat().st_size != size:
            raise SystemExit(f"Damaged model file: {target / name}")


def check(stage):
    """The unzipped app must import everything from its own folder, ignoring this shell."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    run(
        stage / "runtime/python.exe", "-I", "-c",
        "import sys, pathlib, torch, cupy, mujoco, aiortc, xr, imgui_bundle, ito.app.__main__\n"
        "from ito.reconstruction.mast3r_runtime import MODELS\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "assert MODELS == root / 'models', MODELS\n"
        "for module in (torch, cupy, mujoco, aiortc, xr, imgui_bundle, ito):\n"
        "    assert pathlib.Path(module.__file__).is_relative_to(root), module.__file__\n"
        "assert torch.version.cuda, 'torch without CUDA'\n",
        stage, env=env, cwd=stage,
    )  # fmt: skip


def archive(stage, path):
    print(f"Writing {path}", flush=True)
    partial = path.with_suffix(".part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                # Weights do not compress; storing them saves minutes and costs nothing.
                stored = file.suffix == ".safetensors"
                bundle.write(
                    file,
                    Path("ito") / file.relative_to(stage),
                    zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                )
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, help="copy a built ito.build bundle")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist", help="output directory")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("The Windows release is built on Windows")
    stage = args.dist / "ito"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="ito-release-", dir=args.dist) as work:
        work = Path(work)
        launcher(stage, work)
        runtime(stage, work)
    models(stage, args.models)
    check(stage)
    archive(stage, args.dist / f"{NAME}.zip")
    size = (args.dist / f"{NAME}.zip").stat().st_size
    print(f"Release ready: {args.dist / NAME}.zip ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
