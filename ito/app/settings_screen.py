import logging
from concurrent.futures import ThreadPoolExecutor

from imgui_bundle import imgui

from ito import diagnostics
from ito.desktop.settings import AppSettings, load, save


class SettingsScreen:
    def __init__(self, layout):
        self.layout = layout
        self.theme = layout.theme
        self.placed = layout.placed
        self.settings = load()
        self.error = self.save_error = None
        self.typed_splats = str(self.settings.max_splats)
        self.saved = self.settings
        self.writes = []
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ito-settings")

    def change(self, **values):
        try:
            selected = AppSettings.model_validate(self.settings.model_dump() | values)
        except ValueError as exc:
            self.error = (
                "Max splats must be between 1 and 2,147,483,647."
                if "max_splats" in values
                else str(exc).splitlines()[0]
            )
            return
        self.settings, self.error = selected, None
        self.typed_splats = str(selected.max_splats)
        self.writes.append((selected, self.writer.submit(self.persist, selected)))

    @staticmethod
    def persist(selected):
        try:
            save(selected)
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not save settings: %s", exc)
            raise
        diagnostics.current().set_enabled(selected.diagnostic_logging)

    def poll(self):
        while self.writes and self.writes[0][1].done():
            selected, future = self.writes.pop(0)
            try:
                future.result()
                self.saved = selected
                self.save_error = None
            except OSError as exc:
                self.save_error = f"Could not save settings: {exc.strerror or exc}"
                if not self.writes:
                    self.settings = self.saved
                    self.typed_splats = str(self.saved.max_splats)

    def close(self):
        self.writer.shutdown(wait=True)

    def draw(self, *, xr_mode=False):
        s = self.settings
        self.theme.heading("MASt3R-SLAM")
        imgui.text("Precision")
        precision_widths = self.layout.row(1, 1)
        for index, value in enumerate(("full", "half")):
            if index:
                imgui.same_line()
            if s.precision == value:
                imgui.push_style_color(imgui.Col_.button, self.theme.color("selected"))
            clicked = self.layout.button(value.title(), value, precision_widths[index])
            if s.precision == value:
                imgui.pop_style_color()
            if clicked and s.precision != value:
                self.change(precision=value)
        imgui.text_disabled("Half: faster, less VRAM; may lose detail.")
        imgui.text("Input resolution")
        steps = (25, 50, 75, 100)
        index = steps.index(s.input_resolution)
        minus, value_width, plus = self.layout.row(1, 1.85, 1)
        if self.layout.button("-##resolution", "resolution minus", minus, disabled=index == 0):
            self.change(input_resolution=steps[index - 1])
        imgui.same_line()
        value = f"{s.input_resolution}%"
        with self.theme.font("numeric"):
            self.layout.button(value, "resolution value", value_width, disabled=True)
        imgui.same_line()
        if self.layout.button("+##resolution", "resolution plus", plus, disabled=index == 3):
            self.change(input_resolution=steps[index + 1])
        imgui.text_disabled("Lower: faster, less VRAM, blurrier. Keeps the whole frame.")
        self.theme.heading("Scene")
        imgui.text("Max splats")
        minus, value_width, plus = self.layout.row(1, 1.85, 1)
        if self.layout.button("-##splats", "splats minus", minus, disabled=s.max_splats == 1):
            self.change(max_splats=max(1, ((s.max_splats - 1) // 100_000) * 100_000))
        imgui.same_line()
        if xr_mode:
            with self.theme.font("numeric"):
                self.layout.button(f"{s.max_splats:,}", "splats value", value_width, disabled=True)
        else:
            with self.theme.font("numeric"), self.layout.field(value_width):
                edited, self.typed_splats = imgui.input_text("##max splats", self.typed_splats)
            self.placed("splats value")
            if edited:
                try:
                    self.change(max_splats=int(self.typed_splats.replace(",", "")))
                except ValueError:
                    self.error = "Max splats must be a positive whole number."
        imgui.same_line()
        if self.layout.button(
            "+##splats", "splats plus", plus, disabled=s.max_splats == (1 << 31) - 1
        ):
            self.change(max_splats=min((1 << 31) - 1, (s.max_splats // 100_000 + 1) * 100_000))
        imgui.text_disabled("Higher: more scene detail, more VRAM and draw time.")
        imgui.text_disabled("The live VRAM manager may keep fewer splats.")
        self.theme.heading("Diagnostics")
        debug = diagnostics.current()
        imgui.text("Diagnostic logging")
        enabled = (
            debug.enabled if debug.override is not None or debug.error else s.diagnostic_logging
        )
        if self.layout.button(
            "On" if enabled else "Off", "diagnostics", disabled=debug.override is not None
        ):
            self.change(diagnostic_logging=not enabled)
        if debug.override is not None:
            imgui.text_disabled("ITO_DEBUG override")
        imgui.text_disabled("On: records timings to disk for troubleshooting.")
        if imgui.is_item_hovered():
            imgui.set_tooltip(str(debug.path))
        if debug.error:
            imgui.text_wrapped(debug.error)
        imgui.separator()
        reset_width, back_width = self.layout.row(3, 1)
        if self.layout.button("Reset to defaults", "reset", reset_width):
            self.change(**AppSettings().model_dump())
        imgui.same_line()
        back = self.layout.button("Back", "back", back_width)
        if self.error or self.save_error:
            imgui.text_colored(self.theme.color("error"), self.error or self.save_error)
            self.placed("settings error")
        return back
