"""GHOST-RITE — GHOST-LINE (designer-6) in cyan, with VESPER ENGINE's (designer-2) orbit circle.

A cyberbrain terminal used as a ritual: the menu sits inside a binding circle of orbits,
divisions and a ring of inscribed code, flanked by two seals. Cold black-blue glass, cyan
phosphor (after the owner's reference gist), amber/crimson kept for safety only.
2-3 px strokes, no scanlines. Pure ImGui draw lists and style alpha, no shaders.

Motion: the menu opens with a ~2 s intro (circle grows from a point, panel fades in, divisions
tick round, the quotes decode, threads drop, seals fade in), played at startup and when the
pilot comes back to the main menu from a session; never between menu pages. After that only
the quote ring turns, one revolution in three minutes. Every panel (menu pages, HUD, prompts)
fades in and out in 0.1 s; the circle stays between menu pages and fades out for a session.
"""

import math

from imgui_bundle import imgui

NAME = "GHOST-RITE"


def rgba(hex_value, alpha=1.0):
    h = hex_value.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


# Palette
VOID = rgba("02070A")          # menu clear colour
GLASS = rgba("051218")         # window body
GLASS_DEEP = rgba("030D12")    # fields
TITLE = rgba("072A33")         # title band
PHOSPHOR = rgba("C4F7FF")      # body text: pale cyan phosphor
PHOSPHOR_HI = rgba("00E5FF")   # focus / accents / checkmarks
PHOSPHOR_DIM = rgba("1F8494")  # borders, separators, engraved lines
COLD = rgba("8AAFBC")          # secondary / disabled text
CONTROL = rgba("0A2C36")       # buttons
CONTROL_HOVER = rgba("10465A")
CONTROL_ACTIVE = rgba("16627A")
AMBER = rgba("FFB454")         # warning

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

_FONTS = {}
BASE = 18.0


def setup(io, here):
    fonts = here / "fonts"
    body = io.fonts.add_font_from_file_ttf(str(fonts / "Rajdhani-SemiBold.ttf"), BASE)
    _merge_japanese(io, fonts)
    numeric = io.fonts.add_font_from_file_ttf(str(fonts / "UbuntuMono-Regular.ttf"), BASE)
    _merge_japanese(io, fonts)
    display = io.fonts.add_font_from_file_ttf(str(fonts / "Michroma-Regular.ttf"), BASE)
    _merge_japanese(io, fonts)
    # M PLUS 1 Code on its own for the drawn glyphs (糸, 糸操り): same geometric strokes as
    # Ubuntu Mono and Michroma, and it covers kana and the JIS kanji, so Ito can show Japanese.
    kanji = io.fonts.add_font_from_file_ttf(str(fonts / "MPLUS1Code-wght.ttf"), BASE)
    _FONTS.update(body=body, numeric=numeric, display=display, kanji=kanji)

    s = imgui.get_style()
    values = dict(
        alpha=1.0, disabled_alpha=0.72,
        window_padding=(18, 10), window_rounding=0.0, window_border_size=1.0,
        window_min_size=(100, 60), window_title_align=(0.0, 0.5),
        child_rounding=0.0, child_border_size=1.0, popup_rounding=0.0, popup_border_size=1.0,
        frame_padding=(12, 6), frame_rounding=0.0, frame_border_size=1.0,
        item_spacing=(10, 5), item_inner_spacing=(8, 6), cell_padding=(10, 5),
        touch_extra_padding=(0, 0), indent_spacing=22.0, columns_min_spacing=8.0,
        scrollbar_size=14.0, scrollbar_rounding=0.0, scrollbar_padding=2.0,
        grab_min_size=14.0, grab_rounding=0.0, log_slider_deadzone=4.0,
        image_rounding=0.0, image_border_size=1.0,
        tab_rounding=0.0, tab_border_size=1.0, tab_bar_border_size=2.0, tab_bar_overline_size=2.0,
        button_text_align=(0.5, 0.5), selectable_text_align=(0.0, 0.5),
        selectable_rounding=0.0, menu_item_rounding=0.0,
        separator_size=2.0, separator_text_border_size=2.0, separator_text_align=(0.0, 0.5),
        separator_text_padding=(0, 4), display_window_padding=(20, 20),
        display_safe_area_padding=(12, 12), docking_separator_size=2.0,
        mouse_cursor_scale=1.0, anti_aliased_lines=True, anti_aliased_fill=True,
        anti_aliased_lines_use_tex=True, curve_tessellation_tol=1.25,
        circle_tessellation_max_error=0.3, input_text_cursor_size=2.0,
        tree_lines_size=2.0, tree_lines_rounding=0.0, drag_drop_target_rounding=0.0,
        drag_drop_target_border_size=2.0, drag_drop_target_padding=3.0,
    )
    for key, value in values.items():
        setattr(s, key, value)
    s.window_menu_button_position = imgui.Dir_.none

    colors = {
        "text": PHOSPHOR, "text_disabled": COLD,
        "window_bg": GLASS, "child_bg": GLASS_DEEP, "popup_bg": rgba("071A22"),
        "border": PHOSPHOR_DIM, "border_shadow": rgba("000000", 0),
        "frame_bg": GLASS_DEEP, "frame_bg_hovered": rgba("0B2F3A"), "frame_bg_active": rgba("11475A"),
        "title_bg": TITLE, "title_bg_active": TITLE, "title_bg_collapsed": GLASS,
        "menu_bar_bg": TITLE,
        "scrollbar_bg": GLASS_DEEP, "scrollbar_grab": PHOSPHOR_DIM,
        "scrollbar_grab_hovered": PHOSPHOR_HI, "scrollbar_grab_active": PHOSPHOR,
        "check_mark": PHOSPHOR_HI, "checkbox_selected_bg": rgba("0C3844"),
        "slider_grab": PHOSPHOR_HI, "slider_grab_active": PHOSPHOR,
        "button": CONTROL, "button_hovered": CONTROL_HOVER, "button_active": CONTROL_ACTIVE,
        "header": CONTROL, "header_hovered": CONTROL_HOVER, "header_active": CONTROL_ACTIVE,
        "separator": PHOSPHOR_DIM, "separator_hovered": PHOSPHOR_HI, "separator_active": PHOSPHOR,
        "resize_grip": PHOSPHOR_DIM, "resize_grip_hovered": PHOSPHOR_HI, "resize_grip_active": PHOSPHOR,
        "input_text_cursor": PHOSPHOR_HI,
        "tab": CONTROL, "tab_hovered": CONTROL_HOVER, "tab_selected": CONTROL_ACTIVE,
        "tab_selected_overline": PHOSPHOR_HI, "tab_dimmed": GLASS_DEEP,
        "tab_dimmed_selected": CONTROL, "tab_dimmed_selected_overline": PHOSPHOR_DIM,
        "docking_preview": rgba("00E5FF", 0.4), "docking_empty_bg": VOID,
        "plot_lines": PHOSPHOR_HI, "plot_lines_hovered": PHOSPHOR,
        "plot_histogram": AMBER, "plot_histogram_hovered": PHOSPHOR,
        "table_header_bg": TITLE, "table_border_strong": PHOSPHOR_DIM,
        "table_border_light": rgba("123C48"), "table_row_bg": rgba("000000", 0),
        "table_row_bg_alt": rgba("00E5FF", 0.05),
        "text_link": PHOSPHOR_HI, "text_selected_bg": rgba("0E6E82", 0.55),
        "tree_lines": PHOSPHOR_DIM,
        "drag_drop_target": PHOSPHOR_HI, "drag_drop_target_bg": rgba("00E5FF", 0.15),
        "unsaved_marker": AMBER,
        "nav_cursor": PHOSPHOR_HI, "nav_windowing_highlight": PHOSPHOR_HI,
        "nav_windowing_dim_bg": rgba("030708", 0.6), "modal_window_dim_bg": rgba("030708", 0.75),
    }
    for name, color in colors.items():
        s.set_color_(getattr(imgui.Col_, name), color)
    return {"heading": display, "title": display, "numeric": numeric}


def _merge_japanese(io, fonts):
    """Merge M PLUS 1 Code into the font just added, so Japanese text never shows as boxes."""
    cfg = imgui.ImFontConfig()
    cfg.merge_mode = True
    io.fonts.add_font_from_file_ttf(str(fonts / "MPLUS1Code-wght.ttf"), BASE, cfg)


def packed(color):
    """For drawing attached to a panel: fades with the panel (ImGui style alpha)."""
    return imgui.get_color_u32(imgui.ImVec4(*color))


def col(hex_value, alpha):
    """For the background circle: its own alpha, independent of the panel fade."""
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba(hex_value, max(0.0, min(1.0, alpha)))))


# The intro clock. Ito calls intro(t) at startup and on returning to the main menu after a
# session. Until it is called everything is drawn settled (so screenshots need no clock).
_INTRO = {"t0": None}
RING_TURN_S = 180.0  # the quote ring turns once every three minutes


def intro(t):
    _INTRO["t0"] = t
    _LEAVE["t0"] = None


# Every panel (main menu, Settings, HUD, prompts) fades in and out this fast; the theme API
# does the panels. The menu's own ~2 s intro comes on top of it.
FADE = 0.1
_LEAVE = {"t0": None}


def screen(old, new, t):
    """Off to a session: the circle fades out with the menu. Between menu pages it stays."""
    _LEAVE["t0"] = None if new == "menu" else t


def _since(t):
    t0 = _INTRO["t0"]
    return 1e9 if t0 is None else max(0.0, t - t0)


def _ramp(e, start, length):
    return min(1.0, max(0.0, (e - start) / length))


def _ease(p):  # ease-out cubic: fast start, soft landing
    return 1 - (1 - p) ** 3


def _noise(*keys):
    v = 2166136261
    for k in keys:
        v = ((v ^ (k & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return v


def frame_begin(draw_list, size, t):
    # The panel fades in once the circle is half open. Style alpha is plain ImGui: widgets,
    # title bar and the panel's own marks all follow it. Settled: 1.0 (HUD, prompts).
    imgui.get_style().alpha = _ease(_ramp(_since(t), 0.45, 0.6))


# The ring carries quotes as lowercase hex byte pairs, never plain text, "//" between them.
# Upper arc: Pico della Mirandola, Oration on the Dignity of Man (1486). Lower arc: the
# Puppet Master, Ghost in the Shell (1995). Both arcs read left to right, like a coin.
QUOTES = (
    "be the free and proud shaper of your own being.",
    "your effort to remain what you are is what limits you.",
)


def _hex(text):
    return " ".join(f"{b:02x}" for b in text.encode())


def _glyph(dl, ch, x, y, size, color):
    """Draw one glyph with its ink box centred on (x, y), not its line box."""
    font = _FONTS["kanji"]
    g = font.get_font_baked(size).find_glyph(ord(ch))
    dl.add_text(font, size, (x - (g.x0 + g.x1) / 2, y - (g.y0 + g.y1) / 2), color, ch)


def _arc_text(dl, font, size, text, cx, cy, radius, centre, upper, color, track=0.0, pick=None):
    """Text along the circle, glyphs turned to the tangent. upper: tops face out, read clockwise
    over the top; lower: tops face in, read anticlockwise under the bottom. Both left to right.
    pick(i, ch) -> (ch, colour) or None lets a caller swap or hide glyphs (monospace only)."""
    baked = font.get_font_baked(size)
    mid = baked.find_glyph(ord("0"))
    ymid = (mid.y0 + mid.y1) / 2           # digits sit centred on the ring line
    total = sum(baked.get_char_advance(ord(c)) + track for c in text) - track
    sign = 1 if upper else -1
    a = centre - sign * total / 2 / radius
    for i, c in enumerate(text):
        adv = baked.get_char_advance(ord(c))
        shown = (c, color) if pick is None else pick(i, c)
        if shown is None:
            a += sign * (adv + track) / radius
            continue
        g = baked.find_glyph(ord(shown[0]))
        if g.is_visible:
            ac = a + sign * (adv / 2) / radius
            ca, sa = math.cos(ac), math.sin(ac)
            px, py = cx + ca * radius, cy + sa * radius
            rx, ry = (-sa, ca) if upper else (sa, -ca)       # reading direction
            dx, dy = (-ca, -sa) if upper else (ca, sa)       # glyph "down"
            def at(lx, ly):
                lx -= adv / 2
                ly -= ymid
                return imgui.ImVec2(px + lx * rx + ly * dx, py + lx * ry + ly * dy)
            dl.prim_reserve(6, 4)
            dl.prim_quad_uv(at(g.x0, g.y0), at(g.x1, g.y0), at(g.x1, g.y1), at(g.x0, g.y1),
                            imgui.ImVec2(g.u0, g.v0), imgui.ImVec2(g.u1, g.v0),
                            imgui.ImVec2(g.u1, g.v1), imgui.ImVec2(g.u0, g.v1), shown[1])
        a += sign * (adv + track) / radius
    return total / radius


def _seal(dl, x, y, r, color, bright, u):
    """A quiet hanko: one thin ring, 糸 at its heart, a single thread through it."""
    w = 1.5 * u
    dl.add_circle((x, y), r, color, 48, w)
    _glyph(dl, "糸", x, y, 30 * u, bright)
    # The thread passes through the seal, broken where the glyph sits.
    dl.add_line((x, y - r - 10 * u), (x, y - r * 0.55), color, w)
    dl.add_line((x, y + r * 0.55), (x, y + r + 10 * u), color, w)


def window_under(draw_list, pos, size, title, kind, t):
    if kind != "menu":
        return
    bg = imgui.get_background_draw_list()
    x, y, w, h = pos.x, pos.y, size.x, size.y
    u = imgui.get_font_size() / BASE
    e = _since(t)  # seconds into the intro; huge once settled
    # Leaving for a session the whole circle fades with the panel; between menu pages it stays.
    k = 1.0 if _LEAVE["t0"] is None else 1 - _ramp(t - _LEAVE["t0"], 0.0, FADE)
    if k <= 0:
        return

    def c(hex_value, alpha):
        return col(hex_value, alpha * k)
    panel = imgui.get_style().alpha
    # The circle belongs to the screen, not the panel: same place and size on every menu
    # page (Ito, Settings), however tall the panel is. Only the bezel and threads follow it.
    vp = imgui.get_main_viewport()
    cx, cy = vp.pos.x + vp.size.x / 2, vp.pos.y + vp.size.y / 2
    r = 0.47 * vp.size.y
    # Faint glass bezel right behind the panel, kept from GHOST-LINE; fades with the panel.
    bg.add_rect((x - 10 * u, y - 10 * u), (x + w + 10 * u, y + h + 10 * u),
                c("1F8494", 0.40 * panel), 0.0, 2 * u)

    # 0.0-1.1 s: the three orbits open from a point at the centre, a beat apart (designer-2).
    seen = _ramp(e, 0.0, 0.15)
    grow = [_ease(_ramp(e, d, 0.9)) for d in (0.0, 0.08, 0.16)]
    for g, rad, a in zip(grow, (r, r + 26 * u, r + 78 * u), (0.65, 0.40, 0.65)):
        if g > 0:
            bg.add_circle((cx, cy), max(1.0, g * rad), c("1F8494", a * seen), 128, 2 * u)
    outer = grow[2]

    # 0.55-1.2 s: twelve divisions tick round clockwise from twelve o'clock, every third a gate.
    for i in range(12):
        p = _ease(_ramp(e, 0.55 + 0.045 * ((i - 9) % 12), 0.15))
        if p <= 0:
            continue
        a = i * math.pi / 6
        gate = i % 3 == 0
        r0 = outer * (r + 36 * u)
        r1 = r0 + p * outer * ((66 if gate else 54) * u - 36 * u)
        bg.add_line((cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                    (cx + math.cos(a) * r1, cy + math.sin(a) * r1),
                    c("00E5FF", 0.55) if gate else c("1F8494", 0.65), (3 if gate else 2) * u)

    # 0.9-1.4 s: two read arcs sweep open on the outer orbit, then drift slowly backwards.
    sweep = _ease(_ramp(e, 0.9, 0.5))
    if sweep > 0:
        back = -t * 2 * math.pi / (RING_TURN_S * 1.5)
        for start in (200, 20):
            a0 = math.radians(start) + back
            bg.path_arc_to((cx, cy), outer * (r + 78 * u), a0, a0 + math.radians(50) * sweep, 32)
            bg.path_stroke(c("00E5FF", 0.55), 3 * u)

    # The quote ring: one quote per half, "//" at the joins, the whole ring turning slowly.
    # 0.8-1.8 s: it decodes from the middle of each quote outwards, digits flickering first.
    font = _FONTS["numeric"]
    ring = r + 13 * u
    lines = [_hex(q) for q in QUOTES]
    fs = 13 * u
    room = math.pi * ring - 6 * fs
    widest = max(font.get_font_baked(fs).get_char_advance(ord("0")) * len(s) for s in lines)
    if widest > room:
        fs *= room / widest  # too long for the half: smaller type, never squeezed together
    adv = font.get_font_baked(fs).get_char_advance(ord("0"))
    n = max(len(s) for s in lines)
    track = min(0.5 * adv, max(0.0, (room - adv * n) / (n - 1)))
    turn = t * 2 * math.pi / RING_TURN_S
    settled = c("4FD8EA", 0.55)
    flicker = c("00E5FF", 0.85)
    step = int(t * 24)

    def decoder(k, length):
        half = max(1.0, (length - 1) / 2)

        def pick(i, ch):
            due = 1.0 + 0.1 * k + 0.6 * abs(i - half) / half
            if e >= due:
                return ch, settled
            if e < due - 0.2 or ch == " ":
                return None
            return "0123456789abcdef"[_noise(i, k, step) % 16], flicker
        return pick

    spans = []
    for q, (line, centre, upper) in enumerate(((lines[0], -math.pi / 2, True),
                                               (lines[1], math.pi / 2, False))):
        spans.append(_arc_text(bg, font, fs, line, cx, cy, ring, centre + turn, upper, settled,
                               track, decoder(q, len(line))))
    # Each "//" sits in the middle of its gap: the quotes differ in length, so the gaps'
    # centres lean off the horizontal axis by a quarter of the difference.
    lean = (spans[0] - spans[1]) / 4
    joins = c("00E5FF", 0.75 * _ramp(e, 1.0, 0.3))
    for side in (lean, math.pi - lean):
        _arc_text(bg, font, fs, "//", cx, cy, ring, side + turn, True, joins)

    # 1.15-1.8 s: marionette threads drop from the outer orbit to the panel, knotted at the end.
    for k, (deg, fx) in enumerate(((-120, 0.16), (-90, 0.5), (-60, 0.84))):
        p = _ease(_ramp(e, 1.15 + 0.1 * k, 0.45)) * (panel > 0)
        if p <= 0:
            continue
        thread = c("00E5FF", 0.28 * panel)  # threads belong to the panel
        a = math.radians(deg)
        top = (cx + math.cos(a) * (r + 78 * u), cy + math.sin(a) * (r + 78 * u))
        end = (x + w * fx, y - 10 * u)
        # The thread passes behind the inscription: cut it where it crosses the text band.
        dx, dy = end[0] - top[0], end[1] - top[1]
        cuts = []
        for rad in (r + 28 * u, r - 2 * u):
            ox, oy = top[0] - cx, top[1] - cy
            qa, qb, qc = dx * dx + dy * dy, 2 * (ox * dx + oy * dy), ox * ox + oy * oy - rad * rad
            disc = max(0.0, qb * qb - 4 * qa * qc)
            cuts.append((-qb - math.sqrt(disc)) / (2 * qa))
        s0, s1 = cuts
        bg.add_line(top, (top[0] + dx * min(s0, p), top[1] + dy * min(s0, p)), thread, 1.0 * u)
        if p > s1:
            bg.add_line((top[0] + dx * s1, top[1] + dy * s1), (top[0] + dx * p, top[1] + dy * p),
                        thread, 1.0 * u)
        if p >= 1:
            bg.add_rect_filled((end[0] - 2 * u, end[1] - 2 * u), (end[0] + 2 * u, end[1] + 2 * u),
                               c("1F8494", 0.65 * panel))

    # 1.5-2.15 s: the twin seals fade in on the horizontal axis, left then right, settling
    # inwards a little. Only when there is room outside the circle.
    sd = r + 78 * u + 70 * u
    if cx - sd - 60 * u < 0:
        return
    for k, side in enumerate((-1, 1)):
        f = _ease(_ramp(e, 1.5 + 0.15 * k, 0.5))
        if f > 0:
            _seal(bg, cx + side * (sd + (1 - f) * 24 * u), cy, 30 * u,
                  c("1F8494", 0.65 * f), c("00E5FF", 0.55 * f), u)


def _stitch(dl, a, b, color, u, on=7.0, off=5.0):
    """A running stitch (sashiko): short dashes along a to b."""
    length = math.dist(a, b)
    if length <= 0:
        return
    dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    s = off * u / 2
    while s < length:
        e = min(s + on * u, length)
        dl.add_line((a[0] + dx * s, a[1] + dy * s), (a[0] + dx * e, a[1] + dy * e), color, 2 * u)
        s = e + off * u


def link_thread(dl, a, b, state, u, t=0.0):
    """The link drawn as a thread from a to b: "taut" (connected), "slack" (lagging),
    "cut" (link lost, 糸が切れた). Ito would call it where it shows link status."""
    (ax, ay), (bx, by) = a, b
    if state == "taut":
        c = packed(PHOSPHOR_HI)
        dl.add_line(a, b, c, 2 * u)
    elif state == "slack":
        c = packed(rgba("1F8494"))
        sag = min(0.18 * abs(bx - ax), 18 * u)
        dl.add_bezier_cubic(a, (ax + (bx - ax) / 3, ay + sag), (ax + 2 * (bx - ax) / 3, by + sag), b, c, 2 * u)
    else:
        c = packed(COLORS["error"])
        mx = (ax + bx) / 2
        # Two loose ends falling away from the break.
        dl.add_bezier_cubic(a, (mx - 40 * u, ay), (mx - 14 * u, ay + 6 * u), (mx - 8 * u, ay + 22 * u), c, 2 * u)
        dl.add_bezier_cubic(b, (mx + 40 * u, by), (mx + 14 * u, by + 6 * u), (mx + 8 * u, by + 22 * u), c, 2 * u)
    for px, py in (a, b):
        dl.add_rect_filled((px - 3 * u, py - 3 * u), (px + 3 * u, py + 3 * u), c)


def window_over(draw_list, pos, size, title, kind, t):
    x, y, w, h = pos.x, pos.y, size.x, size.y
    u = imgui.get_font_size() / BASE
    draw_list.push_clip_rect((x, y), (x + w, y + h), False)
    if kind == "prompt":
        # Reticle brackets in the prompt's border colour: unmistakable, over the safety panel edge.
        c = packed(COLORS["prompt_border"])
        length = 18 * u
        for xx, yy, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            draw_list.add_line((xx + sx * 3, yy + sy * 3), (xx + sx * length, yy + sy * 3), c, 3 * u)
            draw_list.add_line((xx + sx * 3, yy + sy * 3), (xx + sx * 3, yy + sy * length), c, 3 * u)
    else:
        hi = packed(PHOSPHOR_HI)
        th = imgui.get_frame_height()
        # Phosphor bar on the left of the title band and a cut-corner mark at its right.
        draw_list.add_rect_filled((x, y), (x + 4 * u, y + th), hi)
        draw_list.add_line((x + w - 22 * u, y + th - 5 * u), (x + w - 8 * u, y + th - 5 * u), hi, 3 * u)
        draw_list.add_line((x + w - 8 * u, y + th - 5 * u), (x + w - 8 * u, y + 6 * u), hi, 3 * u)
        # Bottom-left corner tab: the panel's "cable in" mark.
        draw_list.add_line((x, y + h - 2 * u), (x + 36 * u, y + h - 2 * u), hi, 3 * u)
        draw_list.add_line((x + 2 * u, y + h), (x + 2 * u, y + h - 24 * u), hi, 3 * u)
        if kind == "menu":
            # Running stitch just inside the border, below the title band.
            st = packed(rgba("1F8494", 0.85))
            i = 6 * u
            top, bot, lft, rgt = y + th + i, y + h - i, x + i, x + w - i
            for a, b in (((lft, top), (rgt, top)), ((rgt, top), (rgt, bot)),
                         ((rgt, bot), (lft + 40 * u, bot)), ((lft, bot - 28 * u), (lft, top))):
                _stitch(draw_list, a, b, st, u)
        elif kind == "hud":
            # The link as a taut thread along the title band (the HUD only draws when connected).
            link_thread(draw_list, (x + 48 * u, y + th / 2), (x + w - 34 * u, y + th / 2), "taut", u, t)
    draw_list.pop_clip_rect()
