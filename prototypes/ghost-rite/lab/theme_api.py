"""What a theme may control, and the few hooks Ito's screens would call to let it.

A theme is a directory holding theme.py (plus its fonts and their licences). theme.py may define:

    NAME = "Short name"

    def setup(io, here: pathlib.Path) -> dict[str, imgui.ImFont]:
        # Called once after the ImGui context exists, before the first frame.
        # Load fonts with io.fonts.add_font_from_file_ttf(str(here / "fonts/x.ttf"), size_px).
        # The FIRST font added becomes the default (body) font.
        # Set the style with imgui.get_style() (colours: style.set_color_(imgui.Col_.x, ImVec4)).
        # Return named fonts for roles: "heading", "numeric", "title" (all optional).

    COLORS = {"error": (r, g, b, a), ...}   # semantic colours Ito hard-codes today, see DEFAULTS

    def frame_begin(draw_list, size, t): ...  # background draw list, before any window
    def window_under(draw_list, pos, size, title, kind, t): ...  # right after begin(): drawn
        # under the widgets. pos/size are last frame's (auto-resized windows settle in 2 frames).
    def window_over(draw_list, pos, size, title, kind, t): ...   # just before end(): drawn over
        # the widgets. kind is "menu", "hud" or "prompt".
    def frame_end(draw_list, size, t): ...    # foreground draw list, after all windows
    def intro(t): ...  # called at startup and when Ito returns to the main menu after a
        # session (never between menu pages); a theme may play an opening from time t on.
        # With fades, it is called when the main menu starts fading in, not at the switch.
    def screen(old, new, t): ...  # Ito switches screens; old/new are kinds, "menu" (main
        # menu, Settings) or "session"; t is when the old screen starts fading out.
    FADE = 0.1  # seconds each panel takes to fade in and to fade out (default 0: instant)

Fades: Ito asks the theme what to draw instead of drawing its screen straight away.

    theme.show("settings", "menu")                # when the screen changes (intro=True after a session)
    for name, alpha, leaving in theme.showing():   # every frame: old screen out, then new one in
        with theme.faded(alpha, leaving):          # leaving: drawn read-only, clicks ignored
            draw(name)
    alpha = theme.panel("stopped", robot_stopped) # a panel that comes and goes within a screen
    if alpha:
        with theme.faded(alpha, not robot_stopped):
            stopped_prompt()

t is Ito's clock in seconds (imgui.get_time()), the same clock in every hook.

Everything except setup is optional. Ito would adopt the same hooks, so keep them cheap
(they run every frame, also in the headset at 90 Hz).
"""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

from imgui_bundle import imgui

# The colours Ito main (ff9993b) hard-codes; a theme overrides any of them in COLORS.
DEFAULTS = {
    "clear": (0.05, 0.06, 0.07, 1),  # menu screens: no scene behind them
    "error": (1, 0.45, 0.4, 1),
    "warning": (1, 0.65, 0.3, 1),
    "safe": (0.85, 0.9, 0.95, 1),  # "E-STOP: not latched"
    "selected": (0.18, 0.42, 0.65, 1),  # the chosen Precision button
    "stop_bg": (0.32, 0.22, 0.02, 0.92),  # "Robot stopped" prompt
    "stop_bg_bright": (0.95, 0.55, 0.05, 0.97),  # same, while the pilot tries to drive
    "estop_bg": (0.55, 0.08, 0.06, 0.92),  # "E-STOP LATCHED" prompt
    "estop_bg_bright": (0.85, 0.12, 0.08, 0.97),
    "prompt_border": (1, 0.8, 0.3, 1),
}


class Theme:
    def __init__(self, directory):
        self.dir = Path(directory).resolve()
        spec = importlib.util.spec_from_file_location(f"theme_{self.dir.name}", self.dir / "theme.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.name = getattr(self.mod, "NAME", self.dir.name)
        self.fonts = {}
        self.t = 0.0
        self.fade = float(getattr(self.mod, "FADE", 0.0))
        self.current = None  # (name, kind)
        self.previous = None
        self.switched = 0.0
        self.pending_intro = False
        self.panels = {}  # name -> (visible, since)

    def setup(self):
        self.fonts = self.mod.setup(imgui.get_io(), self.dir) or {}

    def color(self, role):
        return getattr(self.mod, "COLORS", {}).get(role, DEFAULTS[role])

    @contextmanager
    def font(self, role, scale=1.0):
        """Ito would call this where it shows headings, numbers or big prompts."""
        font = self.fonts.get(role)
        if font is None and scale == 1.0:
            yield
            return
        imgui.push_font(font, imgui.get_font_size() * scale)
        try:
            yield
        finally:
            imgui.pop_font()

    def intro(self):
        """Ito would call this at startup and on returning to the main menu from a session."""
        hook = getattr(self.mod, "intro", None)
        if hook:
            hook(self.t)

    def show(self, name, kind, intro=False):
        """Ito calls this when it changes screen. kind: "menu" or "session". intro: play the
        theme's opening as the new screen comes in (startup, back from a session)."""
        if self.current is not None and self.current[0] == name:
            return
        hook = getattr(self.mod, "screen", None)
        if hook:
            hook(self.current[1] if self.current else None, kind, self.t)
        self.previous = self.current if self.fade > 0 else None
        self.current = (name, kind)
        self.switched = self.t
        self.pending_intro = intro

    def showing(self):
        """What to draw this frame: [(name, alpha, leaving)]. The old screen fades out first,
        then the new one fades in, so two panels never overlap."""
        if self.current is None:
            return []
        e = self.t - self.switched
        out = self.fade if self.previous is not None else 0.0
        if e < out:
            return [(self.previous[0], 1 - e / out, True)]
        if self.pending_intro:
            self.pending_intro = False
            self.intro_at(self.switched + out)
        p = 1.0 if self.fade <= 0 else min(1.0, (e - out) / self.fade)
        return [(self.current[0], p, False)]

    def panel(self, name, visible):
        """Alpha for a panel that comes and goes within a screen (0: don't draw it)."""
        state = self.panels.get(name)
        if state is None:
            self.panels[name] = state = (False, -1e9)  # first sight fades in too
        if state[0] != visible:
            # Turning round mid-fade starts from where it was, not from scratch.
            a = self._panel_alpha(state)
            since = self.t - (a if visible else 1 - a) * self.fade
            self.panels[name] = state = (visible, since)
        return self._panel_alpha(state)

    def _panel_alpha(self, state):
        visible, since = state
        p = 1.0 if self.fade <= 0 else min(1.0, max(0.0, (self.t - since) / self.fade))
        return p if visible else 1 - p

    @contextmanager
    def faded(self, alpha, leaving=False):
        """Draw a screen or panel at alpha (on top of the theme's own style alpha). leaving:
        it is on its way out, so its widgets take no input (and look the same)."""
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

    def intro_at(self, t):
        hook = getattr(self.mod, "intro", None)
        if hook:
            hook(t)

    def frame_begin(self, size):
        hook = getattr(self.mod, "frame_begin", None)
        if hook:
            hook(imgui.get_background_draw_list(), size, self.t)

    def window(self, title, kind, when):
        hook = getattr(self.mod, f"window_{when}", None)
        if hook:
            hook(imgui.get_window_draw_list(), imgui.get_window_pos(), imgui.get_window_size(),
                 title, kind, self.t)

    def frame_end(self, size):
        hook = getattr(self.mod, "frame_end", None)
        if hook:
            hook(imgui.get_foreground_draw_list(), size, self.t)
