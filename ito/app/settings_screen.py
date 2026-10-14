"""The same button-only settings page in the desktop window and VR panel."""

from imgui_bundle import imgui

from ito import diagnostics
from ito.desktop.settings import AppSettings, load, save
from ito.reconstruction.slam_image import input_size


class SettingsScreen:
    def __init__(self, placed, camera=None):
        self.placed, self.camera = placed, camera
        self.settings = load()
        self.error = None
        self.typed_splats = str(self.settings.max_splats)

    def change(self, **values):
        try:
            selected = AppSettings.model_validate(self.settings.model_dump() | values)
            save(selected)
        except (OSError, ValueError) as exc:
            self.error = f"Could not save settings: {exc}"
            return
        self.settings, self.error = selected, None
        self.typed_splats = str(selected.max_splats)
        diagnostics.current().set_enabled(selected.diagnostic_logging)

    def button(self, label, name, size=(94, 40), disabled=False):
        imgui.begin_disabled(disabled)
        selected = imgui.button(label, size)
        self.placed(name)
        imgui.end_disabled()
        return selected

    def draw(self, *, xr_mode=False):
        s = self.settings
        imgui.separator_text("MASt3R-SLAM")
        imgui.text("Precision")
        for index, value in enumerate(("full", "half")):
            if index:
                imgui.same_line()
            if s.precision == value:
                imgui.push_style_color(imgui.Col_.button, (0.18, 0.42, 0.65, 1))
            clicked = self.button(value.title(), value, (180, 40))
            if s.precision == value:
                imgui.pop_style_color()
            if clicked and s.precision != value:
                self.change(precision=value)
        imgui.text_disabled("Half: faster, less VRAM; may lose detail.")
        imgui.text("Input resolution")
        steps = (25, 50, 75, 100)
        index = steps.index(s.input_resolution)
        if self.button("-##resolution", "resolution minus", disabled=index == 0):
            self.change(input_resolution=steps[index - 1])
        imgui.same_line()
        value = f"{s.input_resolution}%"
        if self.camera:
            width, height = input_size(self.camera.width, self.camera.height, s.input_resolution)
            value += f" ({width} × {height})"
        self.button(value, "resolution value", (172, 40), True)
        imgui.same_line()
        if self.button("+##resolution", "resolution plus", disabled=index == 3):
            self.change(input_resolution=steps[index + 1])
        imgui.text_disabled("Lower: faster, less VRAM, blurrier. Keeps the whole frame.")
        imgui.separator_text("Scene")
        imgui.text("Max splats")
        if self.button("-##splats", "splats minus", disabled=s.max_splats == 1):
            self.change(max_splats=max(1, s.max_splats // 2))
        imgui.same_line()
        if xr_mode:
            self.button(f"{s.max_splats:,}", "splats value", (172, 40), True)
        else:
            imgui.set_next_item_width(172)
            edited, self.typed_splats = imgui.input_text("##max splats", self.typed_splats)
            self.placed("splats value")
            if edited:
                try:
                    self.change(max_splats=int(self.typed_splats.replace(",", "")))
                except ValueError:
                    self.error = "Max splats must be a positive whole number."
        imgui.same_line()
        if self.button("+##splats", "splats plus", disabled=s.max_splats == (1 << 31) - 1):
            self.change(max_splats=min((1 << 31) - 1, s.max_splats * 2))
        imgui.text_disabled("Higher: more scene detail, more VRAM and draw time.")
        imgui.text_disabled("The live VRAM manager may keep fewer splats.")
        imgui.separator_text("Diagnostics")
        debug = diagnostics.current()
        imgui.text("Diagnostic logging")
        enabled = debug.enabled if debug.override is not None else s.diagnostic_logging
        if self.button(
            "On" if enabled else "Off", "diagnostics", (180, 40), debug.override is not None
        ):
            self.change(diagnostic_logging=not enabled)
        imgui.same_line()
        imgui.text_disabled("ITO_DEBUG override" if debug.override is not None else "")
        imgui.text_disabled("On: records timings to disk for troubleshooting.")
        if imgui.is_item_hovered():
            imgui.set_tooltip(str(debug.path))
        if debug.error:
            imgui.text_wrapped(debug.error)
        imgui.separator()
        if self.button("Reset to defaults", "reset", (270, 40)):
            self.change(**AppSettings().model_dump())
        imgui.same_line()
        back = self.button("Back", "back", (94, 40))
        if self.error:
            imgui.text_colored((1, 0.45, 0.4, 1), self.error)
        return back
