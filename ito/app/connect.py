"""Shown when ito starts without an address: a recent robot, a typed address, or the simulator.

A robot that does not know this pilot brings them back here to type its pairing code, once.
"""

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pygame
from imgui_bundle import imgui

from ito.link.pairing import DIGITS, normalize
from ito.protocol import Credential

from . import settings

# Where the last frame drew each control, so e2e scripts click what the pilot sees.
layout = {}


@dataclass(frozen=True)
class Choice:
    address: str | None  # None pilots the bundled simulated robot.
    xr: bool = False
    code: str | None = None  # The robot's pairing code, when the pilot just typed it.
    credential: Credential | None = None  # What the robot gave this pilot when it paired.
    runtime: str | None = None  # The OpenXR runtime's manifest; None keeps the system's.


@dataclass(frozen=True)
class Runtime:
    name: str
    manifest: str  # As XR_RUNTIME_JSON takes it.


# A runtime chosen when ito was started; the pilot's choice later overrides the variable.
LAUNCH_RUNTIME = os.environ.get("XR_RUNTIME_JSON")


def placed(name):
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    layout[name] = ((low.x + high.x) / 2, (low.y + high.y) / 2)


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
    # Virtual Desktop makes its own runtime the active one, and SteamVR registers itself
    # only when asked to; Steam Link needs SteamVR, so look in every Steam library.
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


def openxr_runtimes():
    """Installed OpenXR runtimes (SteamVR, Virtual Desktop/VDXR, Monado), the active first."""
    manifests = [LAUNCH_RUNTIME] if LAUNCH_RUNTIME else []
    if os.name == "nt":
        manifests += _windows_manifests()
    else:
        config = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
        roots = [config, *os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")]
        manifests += [Path(root) / "openxr/1/active_runtime.json" for root in roots]
        steamvr = "steamapps/common/SteamVR/steamxr_linux64.json"
        manifests.append(Path.home() / ".local/share/Steam" / steamvr)
    runtimes, seen = [], set()
    for manifest in manifests:
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


def choose(window, *, xr=False, error=None, pairing=None):
    """Draw until the pilot picks a robot; None when they close the window.

    With pairing set to an address, first ask for that robot's code; error says why.
    """
    recent = settings.recent()
    address = pairing or (recent[0]["address"] if recent else "")
    code, focus = "", True
    runtimes = openxr_runtimes()
    headset = bool(runtimes)
    xr = xr and headset
    remembered = settings.xr_runtime()
    runtime = next((r for r in runtimes if r.manifest == remembered), None)
    runtime = runtime or (runtimes[0] if runtimes else None)
    window.input.capture(False)
    pygame.display.set_caption("Ito")
    clock = pygame.time.Clock()
    while True:
        events = pygame.event.get()
        if any(e.type == pygame.QUIT for e in events):
            return None
        size = pygame.display.get_window_size()
        window.overlay.begin(events, size, False)
        imgui.set_next_window_pos((size[0] / 2, size[1] / 2), imgui.Cond_.always, (0.5, 0.5))
        imgui.begin(
            "Ito",
            flags=imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse,
        )
        choice = None
        layout.clear()
        if pairing:
            imgui.text(f"Pairing code for {pairing}")
            imgui.text_disabled("The robot's driver printed it when it was set up;")
            imgui.text_disabled("its --show-code option prints it again.")
            if focus:
                imgui.set_keyboard_focus_here()
                focus = False
            imgui.set_next_item_width(300)
            entered, code = imgui.input_text_with_hint(
                "##code", "123 456", code, imgui.InputTextFlags_.enter_returns_true
            )
            placed("code")
            imgui.same_line()
            if imgui.button("Pair") or entered:
                if normalize(code):
                    choice = Choice(pairing, xr, normalize(code))
                else:
                    error = f"A pairing code is {DIGITS} digits"
            placed("pair")
            if imgui.button("Back", (380, 0)):
                pairing = error = None
            placed("back")
        else:
            imgui.text("Robot address")
            imgui.set_next_item_width(300)
            entered, address = imgui.input_text_with_hint(
                "##address", "192.168.1.20:8080", address, imgui.InputTextFlags_.enter_returns_true
            )
            placed("address")
            imgui.same_line()
            if (imgui.button("Connect") or entered) and address.strip():
                typed = address.strip()
                choice = Choice(typed, xr, credential=settings.credential(typed))
            placed("connect")
            if recent:
                imgui.separator_text("Recent robots")
                for index, entry in enumerate(recent):
                    if imgui.button(f"{entry['name']}  {entry['address']}##{index}", (380, 0)):
                        choice = Choice(entry["address"], xr, credential=entry["credential"])
                    placed(f"recent {index}")
            imgui.separator()
            if imgui.button("Try simulated robot", (380, 0)):
                choice = Choice(None, xr)
            placed("simulated")
        if headset:
            _, xr = imgui.checkbox("Pilot in VR headset", xr)
            if xr and len(runtimes) > 1:
                # Steam Link streams through SteamVR, Virtual Desktop through its own runtime.
                imgui.set_next_item_width(300)
                if imgui.begin_combo("Runtime", runtime.name):
                    for index, option in enumerate(runtimes):
                        if imgui.selectable(f"{option.name}##{index}", option == runtime)[0]:
                            runtime = option
                    imgui.end_combo()
                placed("runtime")
        window.overlay.diagnostic_controls()
        if error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 380)
            imgui.text_colored((1, 0.45, 0.4, 1), error)
            imgui.pop_text_wrap_pos()
        imgui.end()
        window.context.screen.use()
        window.context.viewport = (0, 0, *size)
        window.context.clear(0.05, 0.06, 0.07)
        window.overlay.render()
        pygame.display.flip()
        if choice:
            layout.clear()
            if choice.xr and runtime:
                settings.remember_xr_runtime(runtime.manifest)
                choice = replace(choice, runtime=runtime.manifest)
            return choice
        clock.tick(30)
