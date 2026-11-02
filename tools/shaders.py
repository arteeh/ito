"""Compile Ito's shaders (ito/shaders/*.vert, .geom, .frag, .comp: GLSL for Vulkan) to SPIR-V
next to them (NAME.spv), with glslang 16.6.0: the one on PATH, or its Khronos release, fetched
once into the user's cache and checked against its hash.

    python tools/shaders.py

release/windows.py runs it before building the wheel."""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "ito" / "shaders"
STAGES = (".vert", ".geom", ".frag", ".comp")
VERSION = "16.6.0"
RELEASES = {
    "linux": ("linux-x86_64", "a3fc4f083b1793eb53e55fa3577ac9649ffbe0340715e30d116290fb5382393f"),
    "win32": ("windows-x86_64", "82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b"),
}


def cache():
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "ito" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "ito"


def glslang() -> Path:
    found = shutil.which("glslang")
    if found:
        version = subprocess.run([found, "--version"], capture_output=True, text=True).stdout
        if f":{VERSION}" in version:
            return Path(found)
    if sys.platform not in RELEASES:
        raise SystemExit(f"glslang {VERSION} is needed on PATH")
    name, digest = RELEASES[sys.platform]
    folder = cache() / f"glslang-{VERSION}"
    exe = folder / "bin" / ("glslang.exe" if sys.platform == "win32" else "glslang")
    if exe.exists():
        return exe
    url = (
        "https://github.com/KhronosGroup/glslang/releases/download/"
        f"{VERSION}/glslang-{VERSION}-{name}-release.zip"
    )
    print(f"fetching {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit(f"{url} does not match its known SHA-256")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extract(f"bin/{exe.name}", folder)
    exe.chmod(0o755)
    return exe


def main():
    compiler = glslang()
    sources = sorted(p for p in SHADERS.iterdir() if p.suffix in STAGES)
    for source in sources:
        target = source.with_name(source.name + ".spv")
        result = subprocess.run(
            [
                compiler,
                "-V",
                "--target-env",
                "vulkan1.3",
                "-I" + str(SHADERS),
                "-o",
                target,
                source,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise SystemExit(f"{source.name}:\n{result.stdout}{result.stderr}")
    print(f"{len(sources)} shaders compiled", flush=True)


if __name__ == "__main__":
    main()
