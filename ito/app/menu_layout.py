"""One centred, font-scaled content width for desktop and headset menu screens."""

from contextlib import contextmanager

from imgui_bundle import imgui


class MenuLayout:
    def __init__(self, placed, theme):
        self.placed = placed
        self.theme = theme
        self.scale = 1.0

    @contextmanager
    def window(self, title, size):
        self.scale = imgui.get_font_size() / 13
        margin = 16 * self.scale
        padding = imgui.get_style().window_padding.x
        width = min(420 * self.scale + 2 * padding, size[0] - 2 * margin)
        imgui.set_next_window_pos((size[0] / 2, size[1] / 2), imgui.Cond_.always, (0.5, 0.5))
        imgui.set_next_window_size_constraints((width, 0), (width, size[1] - 2 * margin))
        with self.theme.font("title"):
            imgui.begin(
                title,
                flags=imgui.WindowFlags_.always_auto_resize
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_collapse,
            )
        self.theme.window(title, "menu", "under")
        imgui.push_text_wrap_pos(0)
        try:
            yield
        finally:
            imgui.pop_text_wrap_pos()
            self.theme.window(title, "menu", "over")
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
        selected = imgui.button(
            label, (self.row(1)[0] if width is None else width, 40 * self.scale)
        )
        self.placed(name)
        imgui.end_disabled()
        return selected
