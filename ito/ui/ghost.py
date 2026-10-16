import math

from imgui_bundle import imgui


def rgba(hex_value, alpha=1.0):
    h = hex_value.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


VOID = rgba("02070A")
GLASS = rgba("051218")
GLASS_DEEP = rgba("030D12")
TITLE = rgba("072A33")
PHOSPHOR = rgba("C4F7FF")
PHOSPHOR_HI = rgba("00E5FF")
PHOSPHOR_DIM = rgba("1F8494")
COLD = rgba("8AAFBC")
CONTROL = rgba("0A2C36")
CONTROL_HOVER = rgba("10465A")
CONTROL_ACTIVE = rgba("16627A")
AMBER = rgba("FFB454")

COLORS = {
    "clear": VOID,
    "error": rgba("FF7A6B"),
    "warning": AMBER,
    "safe": rgba("B8EEF5"),
    "selected": rgba("0E6E82"),
    "stop_bg": rgba("3D2A0E", 0.97),
    "stop_bg_bright": rgba("8A5A12", 0.98),
    "estop_bg": rgba("5A0F14", 0.97),
    "estop_bg_bright": rgba("A0151C", 0.98),
    "prompt_border": rgba("FFD27A"),
}

BASE = 18.0


def _merge_japanese(io, fonts, destinations):
    for destination in destinations:
        cfg = imgui.ImFontConfig()
        cfg.merge_mode = True
        cfg.dst_font = destination
        io.fonts.add_font_from_file_ttf(str(fonts / "MPLUS1Code-wght.ttf"), BASE, cfg)


def packed(color):
    return imgui.get_color_u32(imgui.ImVec4(*color))


def col(hex_value, alpha):
    return imgui.color_convert_float4_to_u32(
        imgui.ImVec4(*rgba(hex_value, max(0.0, min(1.0, alpha))))
    )


RING_TURN_S = 180.0  # the quote ring turns once every three minutes


FADE = 0.1


def _ramp(e, start, length):
    return min(1.0, max(0.0, (e - start) / length))


def _ease(p):
    return 1 - (1 - p) ** 3


def _noise(*keys):
    v = 2166136261
    for k in keys:
        v = ((v ^ (k & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return v


# The ring carries quotes as lowercase hex byte pairs, never plain text, "//" between them.
# Upper arc: Pico della Mirandola, Oration on the Dignity of Man (1486). Lower arc: the
# Puppet Master, Ghost in the Shell (1995). All glyphs stand with their tops outwards and run
# clockwise, so whichever stretch of the turning ring is at the top reads upright.
QUOTES = (
    "be the free and proud shaper of your own being.",
    "your effort to remain what you are is what limits you.",
)


def _hex(text):
    return " ".join(f"{b:02x}" for b in text.encode())


def link_thread(dl, a, b, state, u, t=0.0):
    (ax, ay), (bx, by) = a, b
    if state == "taut":
        c = packed(PHOSPHOR_HI)
        dl.add_line(a, b, c, 2 * u)
    elif state == "slack":
        c = packed(rgba("1F8494"))
        sag = min(0.18 * abs(bx - ax), 18 * u)
        dl.add_bezier_cubic(
            a, (ax + (bx - ax) / 3, ay + sag), (ax + 2 * (bx - ax) / 3, by + sag), b, c, 2 * u
        )
    else:
        c = packed(COLORS["error"])
        mx = (ax + bx) / 2
        dl.add_bezier_cubic(
            a, (mx - 40 * u, ay), (mx - 14 * u, ay + 6 * u), (mx - 8 * u, ay + 22 * u), c, 2 * u
        )
        dl.add_bezier_cubic(
            b, (mx + 40 * u, by), (mx + 14 * u, by + 6 * u), (mx + 8 * u, by + 22 * u), c, 2 * u
        )
    for px, py in (a, b):
        dl.add_rect_filled((px - 3 * u, py - 3 * u), (px + 3 * u, py + 3 * u), c)


HEX_QUOTES = tuple(_hex(quote) for quote in QUOTES)


class GhostRite:
    def __init__(self):
        self.fonts = {}
        self.intro_started = None
        self.leaving = None
        self.vertices = tuple(imgui.ImVec2() for _ in range(4))
        self.uv = tuple(imgui.ImVec2() for _ in range(4))

    def setup(self, io, here):
        fonts = here / "fonts"
        body = io.fonts.add_font_from_file_ttf(str(fonts / "Rajdhani-SemiBold.ttf"), BASE)
        numeric = io.fonts.add_font_from_file_ttf(str(fonts / "UbuntuMono-Regular.ttf"), BASE)
        display = io.fonts.add_font_from_file_ttf(str(fonts / "Michroma-Regular.ttf"), BASE)
        _merge_japanese(io, fonts, (body, numeric, display))
        kanji = io.fonts.add_font_from_file_ttf(str(fonts / "MPLUS1Code-wght.ttf"), BASE)
        self.fonts.update(body=body, numeric=numeric, display=display, kanji=kanji)

        s = imgui.get_style()
        values = dict(
            alpha=1.0,
            disabled_alpha=0.72,
            window_padding=(18, 10),
            window_rounding=0.0,
            window_border_size=1.0,
            window_min_size=(100, 60),
            window_title_align=(0.0, 0.5),
            child_rounding=0.0,
            child_border_size=1.0,
            popup_rounding=0.0,
            popup_border_size=1.0,
            frame_padding=(12, 6),
            frame_rounding=0.0,
            frame_border_size=1.0,
            item_spacing=(10, 5),
            item_inner_spacing=(8, 6),
            cell_padding=(10, 5),
            touch_extra_padding=(0, 0),
            indent_spacing=22.0,
            columns_min_spacing=8.0,
            scrollbar_size=14.0,
            scrollbar_rounding=0.0,
            scrollbar_padding=2.0,
            grab_min_size=14.0,
            grab_rounding=0.0,
            log_slider_deadzone=4.0,
            image_rounding=0.0,
            image_border_size=1.0,
            tab_rounding=0.0,
            tab_border_size=1.0,
            tab_bar_border_size=2.0,
            tab_bar_overline_size=2.0,
            button_text_align=(0.5, 0.5),
            selectable_text_align=(0.0, 0.5),
            selectable_rounding=0.0,
            menu_item_rounding=0.0,
            separator_size=2.0,
            separator_text_border_size=2.0,
            separator_text_align=(0.0, 0.5),
            separator_text_padding=(0, 4),
            display_window_padding=(20, 20),
            display_safe_area_padding=(12, 12),
            docking_separator_size=2.0,
            mouse_cursor_scale=1.0,
            anti_aliased_lines=True,
            anti_aliased_fill=True,
            anti_aliased_lines_use_tex=True,
            curve_tessellation_tol=1.25,
            circle_tessellation_max_error=0.3,
            input_text_cursor_size=2.0,
            tree_lines_size=2.0,
            tree_lines_rounding=0.0,
            drag_drop_target_rounding=0.0,
            drag_drop_target_border_size=2.0,
            drag_drop_target_padding=3.0,
        )
        for key, value in values.items():
            setattr(s, key, value)
        s.window_menu_button_position = imgui.Dir_.none

        colors = {
            "text": PHOSPHOR,
            "text_disabled": COLD,
            "window_bg": GLASS,
            "child_bg": GLASS_DEEP,
            "popup_bg": rgba("071A22"),
            "border": PHOSPHOR_DIM,
            "border_shadow": rgba("000000", 0),
            "frame_bg": GLASS_DEEP,
            "frame_bg_hovered": rgba("0B2F3A"),
            "frame_bg_active": rgba("11475A"),
            "title_bg": TITLE,
            "title_bg_active": TITLE,
            "title_bg_collapsed": GLASS,
            "menu_bar_bg": TITLE,
            "scrollbar_bg": GLASS_DEEP,
            "scrollbar_grab": PHOSPHOR_DIM,
            "scrollbar_grab_hovered": PHOSPHOR_HI,
            "scrollbar_grab_active": PHOSPHOR,
            "check_mark": PHOSPHOR_HI,
            "checkbox_selected_bg": rgba("0C3844"),
            "slider_grab": PHOSPHOR_HI,
            "slider_grab_active": PHOSPHOR,
            "button": CONTROL,
            "button_hovered": CONTROL_HOVER,
            "button_active": CONTROL_ACTIVE,
            "header": CONTROL,
            "header_hovered": CONTROL_HOVER,
            "header_active": CONTROL_ACTIVE,
            "separator": PHOSPHOR_DIM,
            "separator_hovered": PHOSPHOR_HI,
            "separator_active": PHOSPHOR,
            "resize_grip": PHOSPHOR_DIM,
            "resize_grip_hovered": PHOSPHOR_HI,
            "resize_grip_active": PHOSPHOR,
            "input_text_cursor": PHOSPHOR_HI,
            "tab": CONTROL,
            "tab_hovered": CONTROL_HOVER,
            "tab_selected": CONTROL_ACTIVE,
            "tab_selected_overline": PHOSPHOR_HI,
            "tab_dimmed": GLASS_DEEP,
            "tab_dimmed_selected": CONTROL,
            "tab_dimmed_selected_overline": PHOSPHOR_DIM,
            "docking_preview": rgba("00E5FF", 0.4),
            "docking_empty_bg": VOID,
            "plot_lines": PHOSPHOR_HI,
            "plot_lines_hovered": PHOSPHOR,
            "plot_histogram": AMBER,
            "plot_histogram_hovered": PHOSPHOR,
            "table_header_bg": TITLE,
            "table_border_strong": PHOSPHOR_DIM,
            "table_border_light": rgba("123C48"),
            "table_row_bg": rgba("000000", 0),
            "table_row_bg_alt": rgba("00E5FF", 0.05),
            "text_link": PHOSPHOR_HI,
            "text_selected_bg": rgba("0E6E82", 0.55),
            "tree_lines": PHOSPHOR_DIM,
            "drag_drop_target": PHOSPHOR_HI,
            "drag_drop_target_bg": rgba("00E5FF", 0.15),
            "unsaved_marker": AMBER,
            "nav_cursor": PHOSPHOR_HI,
            "nav_windowing_highlight": PHOSPHOR_HI,
            "nav_windowing_dim_bg": rgba("030708", 0.6),
            "modal_window_dim_bg": rgba("030708", 0.75),
        }
        for name, color in colors.items():
            s.set_color_(getattr(imgui.Col_, name), color)
        return {"heading": display, "title": display, "numeric": numeric}

    def intro(self, t):
        self.intro_started = t
        self.leaving = None

    def screen(self, old, new, t):
        self.leaving = None if new == "menu" else t

    def _since(self, t):
        t0 = self.intro_started
        return 1e9 if t0 is None else max(0.0, t - t0)

    def frame_begin(self, draw_list, size, t):
        imgui.get_style().alpha = _ease(_ramp(self._since(t), 0.45, 0.6))

    def prepare_ring(self, size, scale):
        radius = 0.47 * size[1] + 13 * scale
        pixels = 13 * scale
        font = self.fonts["numeric"]
        room = math.pi * radius - 6 * pixels
        widest = max(
            font.get_font_baked(pixels).get_char_advance(ord("0")) * len(line)
            for line in HEX_QUOTES
        )
        if widest > room:
            pixels *= room / widest
        baked = font.get_font_baked(pixels)
        # Loading glyphs may move ImGui's glyph vector. Retain references only after all
        # inscription characters exist, so cache entries cannot point into the old storage.
        for char in "0123456789abcdef /":
            baked.find_glyph(ord(char))
        self.ring_baked = baked
        self.ring_glyphs = {char: baked.find_glyph(ord(char)) for char in "0123456789abcdef /"}
        self.ring_advance = {char: baked.get_char_advance(ord(char)) for char in self.ring_glyphs}
        advance = self.ring_advance["0"]
        count = max(map(len, HEX_QUOTES))
        self.ring_track = min(0.5 * advance, max(0.0, (room - advance * count) / (count - 1)))
        self.ring_size = pixels
        spans = tuple(
            (sum(self.ring_advance[c] + self.ring_track for c in line) - self.ring_track) / radius
            for line in HEX_QUOTES
        )
        lean = (spans[0] - spans[1]) / 4
        self.inscriptions = (
            self._shape_arc(HEX_QUOTES[0], radius, -math.pi / 2, True, 0),
            self._shape_arc(HEX_QUOTES[1], radius, math.pi / 2, True, 1),
            self._shape_arc("//", radius, lean, True),
            self._shape_arc("//", radius, math.pi - lean, True),
        )
        self.seal_baked = self.fonts["kanji"].get_font_baked(30 * scale)
        self.seal_glyph = self.seal_baked.find_glyph(ord("糸"))

    def _glyph(self, dl, ch, x, y, size, color):
        font = self.fonts["kanji"]
        g = self.seal_glyph
        dl.add_text(font, size, (x - (g.x0 + g.x1) / 2, y - (g.y0 + g.y1) / 2), color, ch)

    def _shape_arc(self, text, radius, centre, upper, quote=None):
        track = self.ring_track if quote is not None else 0.0
        total = sum(self.ring_advance[c] + track for c in text) - track
        sign = 1 if upper else -1
        angle = centre - sign * total / 2 / radius
        half = max(1.0, (len(text) - 1) / 2)
        shaped = []
        for index, char in enumerate(text):
            advance = self.ring_advance[char]
            at = angle + sign * advance / 2 / radius
            ca, sa = math.cos(at), math.sin(at)
            rx, ry = (-sa, ca) if upper else (sa, -ca)
            dx, dy = (-ca, -sa) if upper else (ca, sa)
            due = 0 if quote is None else 1.0 + 0.1 * quote + 0.6 * abs(index - half) / half
            shaped.append((char, due, ca * radius, sa * radius, rx, ry, dx, dy, advance))
            angle += sign * (advance + track) / radius
        return tuple(shaped)

    def _arc_text(self, dl, shaped, cx, cy, turn, color, elapsed, step, quote=0, flicker=0):
        ca, sa = math.cos(turn), math.sin(turn)
        mid = self.ring_glyphs["0"]
        ymid = (mid.y0 + mid.y1) / 2
        p0, p1, p2, p3 = self.vertices
        uv0, uv1, uv2, uv3 = self.uv
        for index, (char, due, px, py, rx, ry, dx, dy, advance) in enumerate(shaped):
            ink = color
            if elapsed < due:
                if elapsed < due - 0.2 or char == " ":
                    continue
                char = "0123456789abcdef"[_noise(index, quote, step) % 16]
                ink = flicker
            glyph = self.ring_glyphs[char]
            if not glyph.is_visible:
                continue
            x0, x1 = glyph.x0 - advance / 2, glyph.x1 - advance / 2
            y0, y1 = glyph.y0 - ymid, glyph.y1 - ymid
            lx, ly = px + x0 * rx + y0 * dx, py + x0 * ry + y0 * dy
            p0.x, p0.y = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            lx, ly = px + x1 * rx + y0 * dx, py + x1 * ry + y0 * dy
            p1.x, p1.y = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            lx, ly = px + x1 * rx + y1 * dx, py + x1 * ry + y1 * dy
            p2.x, p2.y = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            lx, ly = px + x0 * rx + y1 * dx, py + x0 * ry + y1 * dy
            p3.x, p3.y = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            uv0.x, uv0.y = glyph.u0, glyph.v0
            uv1.x, uv1.y = glyph.u1, glyph.v0
            uv2.x, uv2.y = glyph.u1, glyph.v1
            uv3.x, uv3.y = glyph.u0, glyph.v1
            dl.prim_reserve(6, 4)
            dl.prim_quad_uv(p0, p1, p2, p3, uv0, uv1, uv2, uv3, ink)

    def _seal(self, dl, x, y, r, color, bright, u):
        w = 1.5 * u
        dl.add_circle((x, y), r, color, 48, w)
        self._glyph(dl, "糸", x, y, 30 * u, bright)
        dl.add_line((x, y - r - 10 * u), (x, y - r * 0.55), color, w)
        dl.add_line((x, y + r * 0.55), (x, y + r + 10 * u), color, w)

    def window_under(self, draw_list, pos, size, title, kind, t):
        if kind != "menu":
            return
        # Cached draw-list quads bypass push_font, so keep their baked atlas entries alive.
        self.ring_baked.last_used_frame = self.seal_baked.last_used_frame = imgui.get_frame_count()
        bg = imgui.get_background_draw_list()
        x, y, w, h = pos.x, pos.y, size.x, size.y
        u = imgui.get_font_size() / BASE
        e = self._since(t)
        fade = 1.0 if self.leaving is None else 1 - _ramp(t - self.leaving, 0.0, FADE)
        if fade <= 0:
            return

        def c(hex_value, alpha):
            return col(hex_value, alpha * fade)

        panel = imgui.get_style().alpha
        vp = imgui.get_main_viewport()
        cx, cy = vp.pos.x + vp.size.x / 2, vp.pos.y + vp.size.y / 2
        r = 0.47 * vp.size.y
        bg.add_rect(
            (x - 10 * u, y - 10 * u),
            (x + w + 10 * u, y + h + 10 * u),
            c("1F8494", 0.40 * panel),
            0.0,
            2 * u,
        )

        seen = _ramp(e, 0.0, 0.15)
        grow = [_ease(_ramp(e, d, 0.9)) for d in (0.0, 0.08, 0.16)]
        for g, rad, a in zip(grow, (r, r + 26 * u, r + 78 * u), (0.65, 0.40, 0.65), strict=True):
            if g > 0:
                bg.add_circle((cx, cy), max(1.0, g * rad), c("1F8494", a * seen), 128, 2 * u)
        outer = grow[2]

        for i in range(12):
            p = _ease(_ramp(e, 0.55 + 0.045 * ((i - 9) % 12), 0.15))
            if p <= 0:
                continue
            a = i * math.pi / 6
            gate = i % 3 == 0
            r0 = outer * (r + 36 * u)
            r1 = r0 + p * outer * ((66 if gate else 54) * u - 36 * u)
            bg.add_line(
                (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                (cx + math.cos(a) * r1, cy + math.sin(a) * r1),
                c("00E5FF", 0.55) if gate else c("1F8494", 0.65),
                (3 if gate else 2) * u,
            )

        sweep = _ease(_ramp(e, 0.9, 0.5))
        if sweep > 0:
            back = -t * 2 * math.pi / (RING_TURN_S * 1.5)
            for start in (200, 20):
                a0 = math.radians(start) + back
                bg.path_arc_to(
                    (cx, cy), outer * (r + 78 * u), a0, a0 + math.radians(50) * sweep, 32
                )
                bg.path_stroke(c("00E5FF", 0.55), 3 * u)

        turn = t * 2 * math.pi / RING_TURN_S
        settled = c("4FD8EA", 0.55)
        flicker = c("00E5FF", 0.85)
        step = int(t * 24)
        self._arc_text(bg, self.inscriptions[0], cx, cy, turn, settled, e, step, 0, flicker)
        self._arc_text(bg, self.inscriptions[1], cx, cy, turn, settled, e, step, 1, flicker)
        joins = c("00E5FF", 0.75 * _ramp(e, 1.0, 0.3))
        self._arc_text(bg, self.inscriptions[2], cx, cy, turn, joins, e, step)
        self._arc_text(bg, self.inscriptions[3], cx, cy, turn, joins, e, step)

        for n, (deg, fx) in enumerate(((-120, 0.16), (-90, 0.5), (-60, 0.84))):
            p = _ease(_ramp(e, 1.15 + 0.1 * n, 0.45)) * (panel > 0)
            if p <= 0:
                continue
            thread = c("00E5FF", 0.28 * panel)
            a = math.radians(deg)
            top = (cx + math.cos(a) * (r + 78 * u), cy + math.sin(a) * (r + 78 * u))
            end = (x + w * fx, y - 10 * u)
            dx, dy = end[0] - top[0], end[1] - top[1]
            cuts = []
            for rad in (r + 28 * u, r - 2 * u):
                ox, oy = top[0] - cx, top[1] - cy
                qa, qb, qc = (
                    dx * dx + dy * dy,
                    2 * (ox * dx + oy * dy),
                    ox * ox + oy * oy - rad * rad,
                )
                disc = max(0.0, qb * qb - 4 * qa * qc)
                cuts.append((-qb - math.sqrt(disc)) / (2 * qa))
            s0, s1 = cuts
            bg.add_line(top, (top[0] + dx * min(s0, p), top[1] + dy * min(s0, p)), thread, 1.0 * u)
            if p > s1:
                bg.add_line(
                    (top[0] + dx * s1, top[1] + dy * s1),
                    (top[0] + dx * p, top[1] + dy * p),
                    thread,
                    1.0 * u,
                )
            if p >= 1:
                bg.add_rect_filled(
                    (end[0] - 2 * u, end[1] - 2 * u),
                    (end[0] + 2 * u, end[1] + 2 * u),
                    c("1F8494", 0.65 * panel),
                )

        sd = r + 78 * u + 70 * u
        if cx - sd - 60 * u < 0:
            return
        for n, side in enumerate((-1, 1)):
            f = _ease(_ramp(e, 1.5 + 0.15 * n, 0.5))
            if f > 0:
                self._seal(
                    bg,
                    cx + side * (sd + (1 - f) * 24 * u),
                    cy,
                    30 * u,
                    c("1F8494", 0.65 * f),
                    c("00E5FF", 0.55 * f),
                    u,
                )

    def window_over(self, draw_list, pos, size, title, kind, t, link="taut"):
        x, y, w, h = pos.x, pos.y, size.x, size.y
        u = imgui.get_font_size() / BASE
        draw_list.push_clip_rect((x, y), (x + w, y + h), False)
        if kind == "prompt":
            c = packed(COLORS["prompt_border"])
            length = 18 * u
            for xx, yy, sx, sy in (
                (x, y, 1, 1),
                (x + w, y, -1, 1),
                (x, y + h, 1, -1),
                (x + w, y + h, -1, -1),
            ):
                draw_list.add_line(
                    (xx + sx * 3, yy + sy * 3), (xx + sx * length, yy + sy * 3), c, 3 * u
                )
                draw_list.add_line(
                    (xx + sx * 3, yy + sy * 3), (xx + sx * 3, yy + sy * length), c, 3 * u
                )
        else:
            hi = packed(PHOSPHOR_HI)
            th = imgui.get_frame_height()
            draw_list.add_rect_filled((x, y), (x + 4 * u, y + th), hi)
            if kind == "hud":
                link_thread(
                    draw_list, (x + 48 * u, y + th / 2), (x + w - 34 * u, y + th / 2), link, u, t
                )
        draw_list.pop_clip_rect()
