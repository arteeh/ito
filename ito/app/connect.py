from dataclasses import dataclass

import pygame
from imgui_bundle import imgui

from ito.link.pairing import DIGITS, normalize
from ito.protocol import Credential
from ito.xr import runtimes

from . import settings
from .menu_layout import MenuLayout

layout = {}


@dataclass(frozen=True)
class Choice:
    address: str | None
    xr: bool = False
    code: str | None = None
    credential: Credential | None = None
    menu: bool = False


def placed(name):
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    layout[name] = ((low.x + high.x) / 2, (low.y + high.y) / 2)


class Menu:
    def __init__(self, window, *, xr=False, error=None, pairing=None):
        from .settings_screen import SettingsScreen

        self.window = window
        self.theme = window.overlay.theme
        self.theme.menu()
        self.choice = None
        self.recent = settings.recent()
        self.address = pairing or (self.recent[0]["address"] if self.recent else "")
        self.code, self.focus = "", True
        self.headset = bool(runtimes.installed())
        self.xr = xr and self.headset
        self.error, self.pairing = error, pairing
        self.page = "main"
        self.pending_choice = None
        self.layout = MenuLayout(placed, self.theme)
        self.settings = SettingsScreen(self.layout)

    def draw(self, size, *, xr_mode=False):
        self.settings.poll()
        name, alpha, leaving = self.theme.showing()
        if name == "hud":
            choice, self.choice = self.choice, None
            if choice:
                self.theme.leaving = True
            return choice
        with self.theme.faded(alpha, leaving):
            with self.layout.window("Settings" if name == "settings" else "Ito", size):
                choice = self.draw_page(xr_mode=xr_mode, page=name, leaving=leaving)
        if not leaving:
            if choice:
                self.choice = choice
                self.theme.show("hud", "session")
            elif self.page != name:
                self.theme.show(self.page, "menu")
        return None

    def draw_page(self, *, xr_mode, page, leaving):
        choice = None
        layout.clear()
        if self.pending_choice and not leaving:
            imgui.text("Saving settings...")
            if self.settings.save_error:
                self.pending_choice = None
                self.page = "settings"
            elif not self.settings.writes:
                choice, self.pending_choice = self.pending_choice, None
        elif page == "settings":
            if self.settings.draw(xr_mode=xr_mode):
                self.page = "main"
        elif self.pairing:
            imgui.text(f"Pairing code for {self.pairing}")
            imgui.text_disabled("The robot's driver printed it when it was set up;")
            imgui.text_disabled("its --show-code option prints it again.")
            if self.focus and not leaving:
                imgui.set_keyboard_focus_here()
                self.focus = False
            field, action = self.layout.row(3, 1)
            with self.layout.field(field):
                entered, self.code = imgui.input_text_with_hint(
                    "##code", "123 456", self.code, imgui.InputTextFlags_.enter_returns_true
                )
            placed("code")
            imgui.same_line()
            if self.layout.button("Pair", "pair", action) or entered:
                if normalize(self.code):
                    choice = Choice(self.pairing, self.xr, normalize(self.code))
                else:
                    self.error = f"A pairing code is {DIGITS} digits"
            if self.layout.button("Back", "back"):
                self.pairing = self.error = None
        else:
            imgui.text("Robot address")
            field, action = self.layout.row(3, 1)
            with self.layout.field(field):
                entered, self.address = imgui.input_text_with_hint(
                    "##address",
                    "192.168.1.20:8080",
                    self.address,
                    imgui.InputTextFlags_.enter_returns_true,
                )
            placed("address")
            imgui.same_line()
            if (
                self.layout.button("Connect", "connect", action) or entered
            ) and self.address.strip():
                typed = self.address.strip()
                choice = Choice(typed, self.xr, credential=settings.credential(typed))
            if self.recent:
                self.theme.heading("Recent robots")
                for index, entry in enumerate(self.recent):
                    if self.layout.button(
                        f"{entry['name']}  {entry['address']}##{index}", f"recent {index}"
                    ):
                        choice = Choice(entry["address"], self.xr, credential=entry["credential"])
            imgui.separator()
            if self.layout.button("Try simulated robot", "simulated"):
                choice = Choice(None, self.xr)
            if self.layout.button("Settings", "settings"):
                self.page = "settings"
            if xr_mode or self.headset:
                if self.layout.button("Use desktop menu" if xr_mode else "Open menu in VR", "vr"):
                    choice = Choice(None, not xr_mode, menu=True)
        if self.error:
            imgui.text_colored(self.theme.color("error"), self.error)
        if choice and self.settings.writes:
            self.pending_choice, choice = choice, None
        return choice


def choose(window, *, xr=False, error=None, pairing=None):
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
            window.context.clear(*menu.theme.color("clear"))
            window.overlay.render()
            pygame.display.flip()
            if choice:
                layout.clear()
                return choice
            clock.tick(window.fps)
    finally:
        menu.settings.close()
