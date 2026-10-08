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
    ito\\THIRD_PARTY_NOTICES.txt   every bundled package's licence texts, gathered at build
"""

import argparse
import email.parser
import json
import os
import re
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
    # SLAM only runs on GPUs the kernels were compiled for (or can JIT from PTX).
    for name in manifest["native"].values():
        found = []
        for kind in ("elf", "ptx"):
            listing = subprocess.run(
                ["cuobjdump", f"--list-{kind}", target / name], capture_output=True, text=True
            ).stdout
            found += [f"{arch} {kind}" for arch in sorted(set(re.findall(r"sm_\d+", listing)))]
        print(f"{name} CUDA code: {', '.join(found) or 'none found'}", flush=True)


LICENCE_FILE = re.compile(
    r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS|ATTRIBUTION|CHECKPOINTS_NOTICE|L?GPL)", re.I
)


def notices(site, models, runtime, output):
    """Write the licence texts of everything the release ships, from what ships with it.

    Each installed distribution contributes its metadata licence and every licence file
    in its dist-info; native libraries vendored into a wheel are listed from its .libs
    folder and SBOMs. Then the Python runtime's and models\\' licence files, verbatim.
    """
    sections = []
    for info in sorted(site.glob("*.dist-info"), key=lambda p: p.name.lower()):
        meta = email.parser.Parser().parsestr(
            (info / "METADATA").read_text(encoding="utf-8", errors="replace"), headersonly=True
        )
        licence = meta.get("License-Expression") or meta.get("License") or ""
        classifiers = [
            c.removeprefix("License :: ")
            for c in meta.get_all("Classifier") or []
            if "License" in c
        ]
        lines = [f"{meta['Name']} {meta['Version']}"]
        lines += [f"Licence: {licence.strip()}"] if licence.strip() else []
        lines += [f"Classifier: {c}" for c in classifiers]
        if meta.get("Home-page"):
            lines.append(f"Home page: {meta['Home-page']}")
        vendored = set()
        for name in (meta.get("Name"), *_top_level(info)):
            folder = site / f"{(name or '').replace('-', '_')}.libs"
            if folder.is_dir():
                vendored.update(p.name for p in folder.iterdir())
        for sbom in sorted((info / "sboms").glob("*.json")):
            try:
                components = json.loads(sbom.read_text(encoding="utf-8")).get("components", [])
            except ValueError:
                continue
            vendored.update(
                f"{c['name']} {c.get('version', '')}".strip()
                for c in components
                if isinstance(c, dict) and c.get("name") and c["name"] != meta["Name"]
            )
        if vendored:
            lines.append("Bundled native libraries: " + ", ".join(sorted(vendored)))
        files = [p for p in sorted(info.rglob("*")) if p.is_file() and _licence_text(p, info)]
        # Some wheels keep their licence inside the package instead (pygame-ce's LGPL.txt).
        files = files or [p for p in _installed(info, site) if LICENCE_FILE.match(p.name)]
        for path in files:
            lines += ["", f"--- {path.relative_to(site).as_posix()} ---", _read(path)]
        if not files:
            lines.append("(The package ships no licence file; see its metadata above.)")
        sections.append("\n".join(lines))
    for root, label in ((runtime, "Python runtime"), (models, "models")):
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and LICENCE_FILE.match(path.name)
                and "site-packages" not in path.parts
            ):
                relative = path.relative_to(root).as_posix()
                sections.append(f"{label}: {relative}\n\n{_read(path)}")
    output.write_text(
        "Third-party software in this Ito release, with the licence texts it ships with.\n\n"
        + ("\n\n" + "=" * 78 + "\n\n").join(sections)
        + "\n",
        encoding="utf-8",
    )
    return len(sections)


def _top_level(info):
    try:
        return (info / "top_level.txt").read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return []


def _installed(info, site):
    try:
        record = (info / "RECORD").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    paths = (site / line.rsplit(",", 2)[0] for line in record if line)
    return sorted(p for p in paths if p.is_file())


def _licence_text(path, info):
    relative = path.relative_to(info)
    return relative.parts[0] == "licenses" or (
        len(relative.parts) == 1 and LICENCE_FILE.match(path.name)
    )


def _read(path):
    return path.read_text(encoding="utf-8", errors="replace").strip()


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
    count = notices(
        stage / "runtime/Lib/site-packages",
        stage / "models",
        stage / "runtime",
        stage / "THIRD_PARTY_NOTICES.txt",
    )
    print(f"THIRD_PARTY_NOTICES.txt: {count} components", flush=True)
    check(stage)
    archive(stage, args.dist / f"{NAME}.zip")
    size = (args.dist / f"{NAME}.zip").stat().st_size
    print(f"Release ready: {args.dist / NAME}.zip ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
