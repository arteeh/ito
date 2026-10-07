"""Shown when ito starts without an address: a recent robot, a typed address, or the simulator."""

import os
from dataclasses import dataclass
from pathlib import Path

import pygame
from imgui_bundle import imgui

from . import settings


@dataclass(frozen=True)
class Choice:
    address: str | None  # None pilots the bundled simulated robot.
    xr: bool = False


def openxr_runtime():
    """An active OpenXR runtime (SteamVR, Virtual Desktop/VDXR, Monado) is registered."""
    if os.environ.get("XR_RUNTIME_JSON"):
        return True
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\OpenXR\1") as key:
                return bool(winreg.QueryValueEx(key, "ActiveRuntime")[0])
        except OSError:
            return False
    config = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    roots = [config, *os.environ.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":")]
    return any((Path(root) / "openxr/1/active_runtime.json").is_file() for root in roots)


def choose(window, *, xr=False, error=None):
    """Draw until the pilot picks a robot; None when they close the window."""
    recent = settings.recent()
    address = recent[0]["address"] if recent else ""
    headset = openxr_runtime()
    xr = xr and headset
    window.input.capture(False)
    pygame.display.set_caption("Ito")
    clock = pygame.time.Clock()
    while True:
        events = pygame.event.get()
        typing = imgui.get_io().want_text_input  # Escape then cancels the edit instead.
        if any(
            e.type == pygame.QUIT
            or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and not typing)
            for e in events
        ):
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
        imgui.text("Robot address")
        imgui.set_next_item_width(300)
        entered, address = imgui.input_text_with_hint(
            "##address", "192.168.1.20:8080", address, imgui.InputTextFlags_.enter_returns_true
        )
        imgui.same_line()
        if (imgui.button("Connect") or entered) and address.strip():
            choice = Choice(address.strip(), xr)
        if recent:
            imgui.separator_text("Recent robots")
            for index, entry in enumerate(recent):
                if imgui.button(f"{entry['name']}  {entry['address']}##{index}", (380, 0)):
                    choice = Choice(entry["address"], xr)
        imgui.separator()
        if imgui.button("Try simulated robot", (380, 0)):
            choice = Choice(None, xr)
        if headset:
            _, xr = imgui.checkbox("Pilot in VR headset", xr)
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
            return choice
        clock.tick(30)
