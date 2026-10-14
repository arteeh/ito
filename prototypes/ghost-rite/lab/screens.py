"""Ito's ImGui screens, copied from Ito main ff9993b with fake data and no robot.

Sources: ito/app/menu_layout.py (MenuLayout), ito/app/connect.py (Menu main page),
ito/app/settings_screen.py (SettingsScreen.draw), ito/desktop/overlay.py (Overlay.draw, live,
desktop; Overlay._stopped_prompt). Widgets, labels, sizes and order match Ito; the only additions
are the theme hooks (THEME.font / color / window), which Ito would adopt with the theme.
Clicks do nothing here.
"""

import math
from contextlib import contextmanager

from imgui_bundle import imgui

THEME = None  # set by shot.py
layout = {}  # where the last frame drew each control (shot.py hovers one)


def placed(name):
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    layout[name] = ((low.x + high.x) / 2, (low.y + high.y) / 2)


def heading(text):
    with THEME.font("heading"):
        imgui.separator_text(text)


class MenuLayout:  # ito/app/menu_layout.py
    def __init__(self):
        self.scale = 1.0

    @contextmanager
    def window(self, title, size):
        self.scale = imgui.get_font_size() / 13
        margin = 16 * self.scale
        padding = imgui.get_style().window_padding.x
        width = min(420 * self.scale + 2 * padding, size[0] - 2 * margin)
        imgui.set_next_window_pos((size[0] / 2, size[1] / 2), imgui.Cond_.always, (0.5, 0.5))
        imgui.set_next_window_size_constraints((width, 0), (width, size[1] - 2 * margin))
        with THEME.font("title"):  # the title bar is drawn in begin()
            imgui.begin(
                title,
                flags=imgui.WindowFlags_.always_auto_resize
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_collapse,
            )
        THEME.window(title, "menu", "under")
        imgui.push_text_wrap_pos(0)
        try:
            yield
        finally:
            imgui.pop_text_wrap_pos()
            THEME.window(title, "menu", "over")
            imgui.end()

    def row(self, *weights):
        width = imgui.get_content_region_avail().x
        width -= imgui.get_style().item_spacing.x * (len(weights) - 1)
        return tuple(width * weight / sum(weights) for weight in weights)

    @contextmanager
    def field(self, width):
        padding = imgui.get_style().frame_padding.x
        imgui.push_style_var(
            imgui.StyleVar_.frame_padding,
            (padding, (40 * self.scale - imgui.get_font_size()) / 2),
        )
        imgui.set_next_item_width(width)
        try:
            yield
        finally:
            imgui.pop_style_var()

    def button(self, label, name, width=None, *, disabled=False):
        imgui.begin_disabled(disabled)
        selected = imgui.button(label, (self.row(1)[0] if width is None else width, 40 * self.scale))
        placed(name)
        imgui.end_disabled()
        return selected


class State:
    address = "192.168.1.20:8080"
    recent = [
        {"name": "Microduck", "address": "192.168.1.20:8080"},
        {"name": "Microduck (sim)", "address": "172.29.91.193:8081"},
    ]
    error = "Could not reach 192.168.1.31:8080: connection refused"
    precision = "half"
    input_resolution = 75
    max_splats = 1_000_000
    typed_splats = "1000000"
    diagnostics = False


def menu(size, state):  # ito/app/connect.py Menu.draw, main page, headset installed
    lay = MenuLayout()
    with lay.window("Ito", size):
        imgui.text("Robot address")
        field, action = lay.row(3, 1)
        with lay.field(field):
            imgui.input_text_with_hint("##address", "192.168.1.20:8080", state.address)
        placed("address")
        imgui.same_line()
        lay.button("Connect", "connect", action)
        heading("Recent robots")
        for index, entry in enumerate(state.recent):
            lay.button(f"{entry['name']}  {entry['address']}##{index}", f"recent {index}")
        imgui.separator()
        lay.button("Try simulated robot", "simulated")
        lay.button("Settings", "settings")
        lay.button("Open menu in VR", "vr")
        if state.error:
            imgui.text_colored(THEME.color("error"), state.error)


def settings(size, state):  # ito/app/settings_screen.py SettingsScreen.draw (desktop)
    lay = MenuLayout()
    with lay.window("Settings", size):
        heading("MASt3R-SLAM")
        imgui.text("Precision")
        widths = lay.row(1, 1)
        for index, value in enumerate(("full", "half")):
            if index:
                imgui.same_line()
            if state.precision == value:
                imgui.push_style_color(imgui.Col_.button, THEME.color("selected"))
            lay.button(value.title(), value, widths[index])
            if state.precision == value:
                imgui.pop_style_color()
        imgui.text_disabled("Half: faster, less VRAM; may lose detail.")
        imgui.text("Input resolution")
        minus, value_width, plus = lay.row(1, 1.85, 1)
        lay.button("-##resolution", "resolution minus", minus)
        imgui.same_line()
        with THEME.font("numeric"):
            lay.button(f"{state.input_resolution}%", "resolution value", value_width, disabled=True)
        imgui.same_line()
        lay.button("+##resolution", "resolution plus", plus)
        imgui.text_disabled("Lower: faster, less VRAM, blurrier. Keeps the whole frame.")
        heading("Scene")
        imgui.text("Max splats")
        minus, value_width, plus = lay.row(1, 1.85, 1)
        lay.button("-##splats", "splats minus", minus)
        imgui.same_line()
        with lay.field(value_width), THEME.font("numeric"):
            imgui.input_text("##max splats", state.typed_splats)
        placed("splats value")
        imgui.same_line()
        lay.button("+##splats", "splats plus", plus)
        imgui.text_disabled("Higher: more scene detail, more VRAM and draw time.")
        imgui.text_disabled("The live VRAM manager may keep fewer splats.")
        heading("Diagnostics")
        imgui.text("Diagnostic logging")
        lay.button("On" if state.diagnostics else "Off", "diagnostics")
        imgui.text_disabled("On: records timings to disk for troubleshooting.")
        imgui.separator()
        reset_width, back_width = lay.row(3, 1)
        lay.button("Reset to defaults", "reset", reset_width)
        imgui.same_line()
        lay.button("Back", "back", back_width)


def hud(size, *, e_stop=False, oom=False, nudged=False, prompt=True):  # ito/desktop/overlay.py, live, desktop
    safety = "E-STOP LATCHED" if e_stop else "E-STOP: not latched"
    imgui.set_next_window_pos((12, 12), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.9)
    with THEME.font("title"):
        imgui.begin(
            "Ito",
            flags=imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse,
        )
    THEME.window("Ito", "hud", "under")
    with THEME.font("numeric"):
        imgui.text("CONNECTED | RTT 23 ms | 90 fps | Microduck 0.17.2")
    imgui.text_colored(THEME.color("error") if e_stop else THEME.color("safe"), safety)
    for label, value, unit in (
        ("Reconstruction", 2.8, "frames/s"),
        ("Capture to visible", 412.0, "ms"),
        ("GPU busy", 63.0, "%"),
    ):
        with THEME.font("numeric"):
            imgui.text(f"{label}: {value:.1f} {unit}")
    with THEME.font("numeric"):
        imgui.text("Splats: 412,906 / 1,000,000 | VRAM 7.4 / 12.0 GiB")
    imgui.text("MASt3R: half (bf16), 75% input")
    if oom:
        imgui.text_colored(THEME.color("error"), "Out of GPU memory: reconstruction is degraded")
        imgui.text_colored(THEME.color("warning"), "fp16 overflow: invalid points discarded")
    imgui.text("WASD move | PgUp/PgDn rise/fall | Home recenter")
    imgui.text("Click scene / Tab for mouse-look")
    imgui.text("Space stop | E e-stop | R resume | F12 capture | Esc release mouse")
    for index, label in enumerate(("Stop", "E-stop", "Resume", "Disconnect")):
        if index:
            imgui.same_line()
        imgui.button(label)
        placed(label.lower())
    imgui.text("audio: opus 48 kHz, 21 ms")
    imgui.checkbox("Mute microphone", False)
    placed("mute_mic")
    imgui.same_line()
    imgui.checkbox("Mute speakers", True)
    imgui.checkbox("Show simulation", True)
    THEME.window("Ito", "hud", "over")
    imgui.end()
    alpha = THEME.panel("stopped", prompt)  # fades in and out with the theme (lab addition)
    if alpha:
        with THEME.faded(alpha, not prompt):
            stopped_prompt(size, e_stop=e_stop, nudged=nudged)


def stopped_prompt(size, *, e_stop, nudged):  # Overlay._stopped_prompt, desktop
    if e_stop:
        title, action = "E-STOP LATCHED", "Press R (or gamepad A) to resume"
    else:
        title, action = "Robot stopped", "Press R (or gamepad A) to drive"
    bright = nudged and math.sin(THEME.t * 4 * math.pi) > -0.3
    role = ("estop_bg" if e_stop else "stop_bg") + ("_bright" if bright else "")
    width, height = size
    imgui.set_next_window_pos((width / 2, height - 24), imgui.Cond_.always, (0.5, 1))
    imgui.push_style_color(imgui.Col_.window_bg, THEME.color(role))
    imgui.push_style_color(imgui.Col_.border, THEME.color("prompt_border") if nudged else (0, 0, 0, 0))
    imgui.push_style_var(imgui.StyleVar_.window_border_size, 3)
    imgui.begin(
        "Robot stopped",
        flags=imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.always_auto_resize
        | imgui.WindowFlags_.no_inputs
        | imgui.WindowFlags_.no_focus_on_appearing
        | imgui.WindowFlags_.no_nav,
    )
    THEME.window("Robot stopped", "prompt", "under")
    with THEME.font("heading", 1.6):
        imgui.text(title)
        imgui.text(action)
    imgui.text("Movement is ignored until you resume.")
    THEME.window("Robot stopped", "prompt", "over")
    imgui.end()
    imgui.pop_style_var()
    imgui.pop_style_color(2)


# name -> (draw(size, state), uses the scene backdrop, control to hover)
SHOTS = {
    "1-menu": (lambda size, s: menu(size, s), False, "connect"),
    "2-settings": (lambda size, s: settings(size, s), False, "half"),
    "3-hud": (lambda size, s: hud(size, nudged=True), True, "resume"),
    "4-hud-alert": (lambda size, s: hud(size, e_stop=True, oom=True), True, None),
}
