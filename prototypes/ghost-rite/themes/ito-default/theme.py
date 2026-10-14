"""Ito main today: ImGui's built-in dark style and its built-in font (ProggyClean, 13 px)."""

from imgui_bundle import imgui

NAME = "Ito default (main ff9993b)"


def setup(io, here):
    io.fonts.add_font_default()
    imgui.style_colors_dark()
    return {}
