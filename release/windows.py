"""Build the Windows release of Ito into DIST\\ito (tools/build.sh runs it on Windows):

    uv run --python 3.12 python release/windows.py --dist DIST [--cache DIR]

A relocatable CPython with the locked dependencies installed (the pda extra too), started by
release/ito.c's ito.exe (runtime\\pythonw.exe -I -m ito.app), Depth Anything 3 Metric's models in
models\\ (downloaded at pinned revisions, checked, kept in --cache), plus THIRD_PARTY_NOTICES.txt.
No freezer: the release runs exactly what runs from source."""

import argparse
import email.parser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Depth Anything 3 Metric (ito/tofmesh/pda.py, Apache-2.0): folder -> (Hugging Face revision,
# {file: sha256}). The folder is the model's name under huggingface.co/depth-anything.
MODELS = {
    "DA3METRIC-LARGE": ("4010e39f3634a45bc60553321fb49fb760bd594e", {
        "README.md": "4fdd25492cf6401d1459483fd829f6abb399bda8218e1a3ef5ad4a66e9aa5a2a",
        "config.json": "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53",
        "model.safetensors": "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776",
    }),
}  # fmt: skip


def run(*command, **options):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **options)


def runtime(stage, work):
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
        "uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--extra", "pda",
        "--format", "pylock.toml", "--output-file", lock,
        cwd=ROOT,
    )  # fmt: skip
    run(sys.executable, ROOT / "tools/shaders.py")
    run("uv", "build", "--wheel", "--out-dir", work, cwd=ROOT)
    (wheel,) = work.glob("ito-*.whl")
    python = target / "python.exe"
    # --target writes into this copy only, never into the interpreter it was copied from.
    for requirement in (["-r", lock], ["--no-deps", wheel]):
        run(
            "uv", "pip", "install", "--python", python, "--target", site,
            "--link-mode", "copy", "--compile-bytecode", *requirement,
        )  # fmt: skip
    unused = [site / "bin", *site.rglob("*.lib")]
    for path in unused + list(site.glob("xr/**/android*")):
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def models(stage, cache):
    """Depth Anything 3 Metric's models into stage\\models, from the cache or downloaded into it."""
    for name, (revision, files) in MODELS.items():
        for file, sha256 in files.items():
            cached = cache / name / revision / file
            if not cached.is_file() or _digest(cached) != sha256:
                cached.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/depth-anything/{name}/resolve/{revision}/{file}"
                print(f"Downloading {url}", flush=True)
                partial = cached.with_suffix(".part")
                with urllib.request.urlopen(url) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out, 1 << 20)
                if _digest(partial) != sha256:
                    partial.unlink()
                    raise SystemExit(f"{url} does not match its sha256")
                partial.replace(cached)
            target = stage / "models" / name / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)


def developer_shell():
    """Take on a Visual Studio x64 developer shell's environment, for cl.exe."""
    if shutil.which("cl"):
        return
    installer = Path(os.environ["ProgramFiles(x86)"], "Microsoft Visual Studio/Installer")
    found = subprocess.run(
        [installer / "vswhere.exe", "-latest", "-products", "*", "-property", "installationPath"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not found:
        raise SystemExit("Visual Studio with C++ (cl.exe) is needed for ito.exe")
    vcvars = Path(found, "VC/Auxiliary/Build/vcvars64.bat")
    listing = subprocess.run(
        f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, check=True
    ).stdout
    os.environ.update(line.split("=", 1) for line in listing.splitlines() if "=" in line)


def launcher(stage, work):
    developer_shell()
    run(
        "cl", "/nologo", "/O2", "/W4", "/WX", "/DUNICODE", ROOT / "release/ito.c",
        f"/Fe{stage / 'ito.exe'}", f"/Fo{work / 'ito.obj'}",
        "/link", "/SUBSYSTEM:WINDOWS", "user32.lib",
        cwd=work,
    )  # fmt: skip


LICENCE_FILE = re.compile(
    r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS|ATTRIBUTION|CHECKPOINTS_NOTICE|L?GPL)", re.I
)


def notices(site, runtime, output):
    sections = [
        f"Depth Anything 3 Metric model {name} (models\\{name}), Apache-2.0, "
        f"https://huggingface.co/depth-anything/{name}\n\n"
        + _read(runtime.parent / "models" / name / "README.md")
        for name in MODELS
    ]
    vendor = site / "ito/_vendor/da3"
    sections.append(_read(vendor / "NOTICE") + "\n\n" + _read(vendor / "LICENSE"))
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
        files = files or [p for p in _installed(info, site) if LICENCE_FILE.match(p.name)]
        for path in files:
            lines += ["", f"--- {path.relative_to(site).as_posix()} ---", _read(path)]
        if not files:
            lines.append("(The package ships no licence file; see its metadata above.)")
        sections.append("\n".join(lines))
    for path in sorted(runtime.rglob("*")):
        if path.is_file() and LICENCE_FILE.match(path.name) and "site-packages" not in path.parts:
            relative = path.relative_to(runtime).as_posix()
            sections.append(f"Python runtime: {relative}\n\n{_read(path)}")
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
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    run(
        stage / "runtime/python.exe", "-I", "-c",
        "import sys, pathlib, aiortc, xr, imgui_bundle, vulkan\n"
        "import ito.app.__main__, ito.tofmesh.worker, ito.tofmesh.render, torch, ito.tofmesh.da3\n"
        "from ito.tofmesh import pda\n"
        "assert pda.missing('large') is None, pda.missing('large')\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "for module in (aiortc, xr, imgui_bundle, ito):\n"
        "    assert pathlib.Path(module.__file__).is_relative_to(root), module.__file__\n",
        stage, env=env, cwd=stage,
    )  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist", help="output directory")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "models", help="downloaded models, kept for reuse"
    )
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
    models(stage, args.cache)
    count = notices(
        stage / "runtime/Lib/site-packages", stage / "runtime", stage / "THIRD_PARTY_NOTICES.txt"
    )
    print(f"THIRD_PARTY_NOTICES.txt: {count} components", flush=True)
    check(stage)
    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"Release ready: {stage} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
