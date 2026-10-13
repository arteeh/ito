"""Installed OpenXR runtimes, so Ito can find the headset on whichever one has it.

A PC often has several: SteamVR (Steam Link, Steam Frame, cabled headsets), Virtual Desktop's
VDXR, Meta's, Monado. The system's active runtime is only the one installed last, and the
headset may be streaming through another, so Ito asks each in turn.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ito.app import settings


@dataclass(frozen=True)
class Runtime:
    name: str
    manifest: str | None  # As XR_RUNTIME_JSON takes it; None is the loader's own choice.


# Set when Ito was started: the pilot pinned one runtime, so only that one is asked.
PINNED = os.environ.get("XR_RUNTIME_JSON")


def _windows_manifests():
    import winreg

    manifests = []
    khronos = r"SOFTWARE\Khronos\OpenXR\1"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, khronos) as key:
            manifests.append(winreg.QueryValueEx(key, "ActiveRuntime")[0])
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, khronos + r"\AvailableRuntimes") as key:
            index = 0
            while True:
                manifest, disabled, _ = winreg.EnumValue(key, index)
                if not disabled:
                    manifests.append(manifest)
                index += 1
    except OSError:
        pass
    # SteamVR registers itself only when asked to, so look in every Steam library.
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            steam = Path(winreg.QueryValueEx(key, "SteamPath")[0])
        folders = (steam / "steamapps" / "libraryfolders.vdf").read_text(errors="replace")
        libraries = [steam] + [
            Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', folders)
        ]
        manifests += [
            lib / "steamapps" / "common" / "SteamVR" / "steamxr_win64.json" for lib in libraries
        ]
    except OSError:
        pass
    return manifests


def _linux_manifests():
    config = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    roots = [config, *os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")]
    manifests = [Path(root) / "openxr/1/active_runtime.json" for root in roots]
    manifests.append(
        Path.home() / ".local/share/Steam/steamapps/common/SteamVR/steamxr_linux64.json"
    )
    return manifests


def installed():
    """Every runtime to ask for a headset: the one that last had it, then the system's active
    one, then the rest. Empty when none is installed."""
    if PINNED:
        manifests = [PINNED]
    else:
        manifests = [settings.xr_runtime()]
        manifests += _windows_manifests() if os.name == "nt" else _linux_manifests()
    runtimes, seen = [], set()
    for manifest in manifests:
        if not manifest:
            continue
        path = Path(manifest)
        try:
            real = os.path.normcase(path.resolve())
            if real in seen or not path.is_file():
                continue
            seen.add(real)
            name = json.loads(path.read_text())["runtime"].get("name") or path.stem
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        runtimes.append(Runtime(str(name), str(manifest)))
    return runtimes
