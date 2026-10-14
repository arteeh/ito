"""Shown when ito starts without an address: a recent robot, a typed address, or the simulator.

A robot that does not know this pilot brings them back here to type its pairing code, once.
"""

from dataclasses import dataclass

import pygame
from imgui_bundle import imgui

from ito.link.pairing import DIGITS, normalize
from ito.protocol import Credential
from ito.xr import runtimes

from . import settings

# Where the last frame drew each control, so e2e scripts click what the pilot sees.
layout = {}


@dataclass(frozen=True)
class Choice:
    address: str | None  # None pilots the bundled simulated robot.
    xr: bool = False
    code: str | None = None  # The robot's pairing code, when the pilot just typed it.
    credential: Credential | None = None  # What the robot gave this pilot when it paired.
    menu: bool = False  # Switch the menu between desktop and headset without connecting.


def placed(name):
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    layout[name] = ((low.x + high.x) / 2, (low.y + high.y) / 2)


class Menu:
    def __init__(self, window, *, xr=False, error=None, pairing=None):
        from .settings_screen import SettingsScreen

        self.window = window
        self.recent = settings.recent()
        self.address = pairing or (self.recent[0]["address"] if self.recent else "")
        self.code, self.focus = "", True
        self.headset = bool(runtimes.installed())
        self.xr = xr and self.headset
        self.error, self.pairing = error, pairing
        self.page = "main"
        self.pending_choice = None
        self.settings = SettingsScreen(placed, getattr(window, "menu_camera", None))

    def draw(self, size, *, xr_mode=False):
        self.settings.poll()
        imgui.set_next_window_pos((size[0] / 2, size[1] / 2), imgui.Cond_.always, (0.5, 0.5))
        imgui.begin(
            "Settings" if self.page == "settings" else "Ito",
            flags=imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse,
        )
        choice = None
        layout.clear()
        if self.pending_choice:
            imgui.text("Saving settings...")
            if self.settings.save_error:
                self.pending_choice = None
                self.page = "settings"
            elif not self.settings.writes:
                choice, self.pending_choice = self.pending_choice, None
        elif self.page == "settings":
            if self.settings.draw(xr_mode=xr_mode):
                self.page = "main"
        elif self.pairing:
            imgui.text(f"Pairing code for {self.pairing}")
            imgui.text_disabled("The robot's driver printed it when it was set up;")
            imgui.text_disabled("its --show-code option prints it again.")
            if self.focus:
                imgui.set_keyboard_focus_here()
                self.focus = False
            imgui.set_next_item_width(300)
            entered, self.code = imgui.input_text_with_hint(
                "##code", "123 456", self.code, imgui.InputTextFlags_.enter_returns_true
            )
            placed("code")
            imgui.same_line()
            if imgui.button("Pair", (94, 40)) or entered:
                if normalize(self.code):
                    choice = Choice(self.pairing, self.xr, normalize(self.code))
                else:
                    self.error = f"A pairing code is {DIGITS} digits"
            placed("pair")
            if imgui.button("Back", (380, 40)):
                self.pairing = self.error = None
            placed("back")
        else:
            imgui.text("Robot address")
            imgui.set_next_item_width(300)
            entered, self.address = imgui.input_text_with_hint(
                "##address",
                "192.168.1.20:8080",
                self.address,
                imgui.InputTextFlags_.enter_returns_true,
            )
            placed("address")
            imgui.same_line()
            if (imgui.button("Connect", (94, 40)) or entered) and self.address.strip():
                typed = self.address.strip()
                choice = Choice(typed, self.xr, credential=settings.credential(typed))
            placed("connect")
            if self.recent:
                imgui.separator_text("Recent robots")
                for index, entry in enumerate(self.recent):
                    if imgui.button(f"{entry['name']}  {entry['address']}##{index}", (380, 40)):
                        choice = Choice(entry["address"], self.xr, credential=entry["credential"])
                    placed(f"recent {index}")
            imgui.separator()
            if imgui.button("Try simulated robot", (380, 40)):
                choice = Choice(None, self.xr)
            placed("simulated")
            if imgui.button("Settings", (380, 40)):
                self.page = "settings"
            placed("settings")
            if xr_mode or self.headset:
                if imgui.button("Use desktop menu" if xr_mode else "Open menu in VR", (380, 40)):
                    choice = Choice(None, not xr_mode, menu=True)
                placed("vr")
        if self.error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 380)
            imgui.text_colored((1, 0.45, 0.4, 1), self.error)
            imgui.pop_text_wrap_pos()
        if choice and self.settings.writes:
            self.pending_choice, choice = choice, None
        imgui.end()
        return choice


def choose(window, *, xr=False, error=None, pairing=None):
    """No robot or reconstruction runs while the pilot chooses their next connection."""
    menu = Menu(window, xr=xr, error=error, pairing=pairing)
    window.input.capture(False)
    pygame.display.set_caption("Ito")
    try:
        if getattr(window, "xr", None) is not None:
            return window.choose(menu)
        clock = pygame.time.Clock()
        while True:
            events = pygame.event.get()
            if any(e.type == pygame.QUIT for e in events):
                return None
            size = pygame.display.get_window_size()
            window.overlay.begin(events, size, False)
            choice = menu.draw(size)
            window.context.screen.use()
            window.context.viewport = (0, 0, *size)
            window.context.clear(0.05, 0.06, 0.07)
            window.overlay.render()
            pygame.display.flip()
            if choice:
                layout.clear()
                return choice
            clock.tick(30)
    finally:
        menu.settings.close()
