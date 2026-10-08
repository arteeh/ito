"""Shown when ito starts without an address: a recent robot, a typed address, or the simulator.

A robot that does not know this pilot brings them back here to type its pairing code, once.
"""

import os
from dataclasses import dataclass
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


def placed(name):
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    layout[name] = ((low.x + high.x) / 2, (low.y + high.y) / 2)


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


def choose(window, *, xr=False, error=None, pairing=None):
    """Draw until the pilot picks a robot; None when they close the window.

    With pairing set to an address, first ask for that robot's code; error says why.
    """
    recent = settings.recent()
    address = pairing or (recent[0]["address"] if recent else "")
    code, focus = "", True
    headset = openxr_runtime()
    xr = xr and headset
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
            return choice
        clock.tick(30)
