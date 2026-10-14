"""One theme and one ImGui clock for desktop windows and headset panels."""

from contextlib import contextmanager
from pathlib import Path

from imgui_bundle import imgui

from .ghost import BASE, COLORS, FADE, GhostRite


class Panel:
    def __init__(self):
        self.visible = False
        self.since = -1e9

    def alpha(self, now):
        progress = min(1.0, max(0.0, (now - self.since) / FADE))
        return progress if self.visible else 1.0 - progress

    def show(self, visible, now):
        if self.visible != visible:
            alpha = self.alpha(now)
            self.since = now - (alpha if visible else 1.0 - alpha) * FADE
            self.visible = visible
        return self.alpha(now)


class Theme:
    def __init__(self):
        self.art = GhostRite()
        self.fonts = self.art.setup(imgui.get_io(), Path(__file__).parent)
        self.t = 0.0
        self.name = self.previous = self.kind = None
        self.switched = 0.0
        self.pending_intro = False
        self.entering_menu = False
        self.leaving = False
        self.prepared = None
        self.stopped = Panel()
        self.hud = Panel()
        self.hud.visible = True

    @staticmethod
    def color(role):
        return COLORS[role]

    def begin(self, size):
        self.t = imgui.get_time()
        if self.entering_menu:
            self.entering_menu = False
            self.switched = self.t
        # Bake the known UI sizes on entry/scale changes, never during a session draw.
        scale = imgui.get_style().font_scale_main * imgui.get_style().font_scale_dpi
        if self.prepared != (size, scale):
            self.prepared = (size, scale)
            for font in self.art.fonts.values():
                for pixels in (BASE * scale, BASE * 1.6 * scale):
                    baked = font.get_font_baked(pixels)
                    for code in range(32, 127):
                        baked.find_glyph(code)
                    baked.find_glyph(ord("糸"))
            self.art.prepare_ring(size, scale)
        if self.pending_intro and self.t >= self.switched + (FADE if self.previous else 0):
            self.pending_intro = False
            self.art.intro(self.switched + (FADE if self.previous else 0))
        imgui.get_style().alpha = 1.0
        if self.kind == "menu":
            self.art.frame_begin(imgui.get_background_draw_list(), size, self.t)

    def menu(self):
        intro = self.kind != "menu"
        self.name = self.previous = None
        self.leaving = False
        self.show("main", "menu")
        self.entering_menu = True
        self.pending_intro = intro
        self.stopped = Panel()
        self.hud.visible, self.hud.since = True, -1e9

    def show(self, name, kind, *, intro=False):
        if name == self.name:
            return
        self.art.screen(self.kind, kind, self.t)
        self.previous = self.name
        self.name, self.kind = name, kind
        self.switched = self.t
        self.pending_intro = intro
        self.leaving = False
        if intro and self.previous is None:
            self.pending_intro = False
            self.art.intro(self.t)

    def showing(self):
        elapsed = self.t - self.switched
        if self.leaving:
            return self.name, max(0.0, 1.0 - elapsed / FADE), True
        if self.previous is not None and elapsed < FADE:
            return self.previous, 1.0 - elapsed / FADE, True
        offset = FADE if self.previous is not None else 0.0
        return self.name, min(1.0, max(0.0, (elapsed - offset) / FADE)), False

    def leave(self):
        if not self.leaving:
            self.switched = self.t
            self.leaving = True
            self.art.screen(self.kind, "session", self.t)

    @contextmanager
    def faded(self, alpha, leaving=False):
        imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * alpha)
        if leaving:
            imgui.push_style_var(imgui.StyleVar_.disabled_alpha, 1.0)
            imgui.begin_disabled()
        try:
            yield
        finally:
            if leaving:
                imgui.end_disabled()
                imgui.pop_style_var()
            imgui.pop_style_var()

    @contextmanager
    def font(self, role, scale=1.0):
        imgui.push_font(self.fonts[role], imgui.get_font_size() * scale)
        try:
            yield
        finally:
            imgui.pop_font()

    def heading(self, text):
        with self.font("heading"):
            imgui.separator_text(text)

    def window(self, title, kind, when, *, link="taut"):
        draw = self.art.window_under if when == "under" else self.art.window_over
        args = (
            imgui.get_window_draw_list(),
            imgui.get_window_pos(),
            imgui.get_window_size(),
            title,
            kind,
            self.t,
        )
        if when == "under":
            draw(*args)
        else:
            draw(*args, link=link)
