import logging
from concurrent.futures import ThreadPoolExecutor

from imgui_bundle import imgui

from ito import diagnostics
from ito.desktop.settings import AppSettings, load, save

MEMORY_STEPS = (2, 3, 5, 7, 10, 14, 20, 28, 40)  # Fewer / More on the settings screen


class SettingsScreen:
    def __init__(self, layout):
        self.layout = layout
        self.theme = layout.theme
        self.placed = layout.placed
        self.settings = load()
        self.error = self.save_error = None
        self.saved = self.settings
        self.writes = []
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ito-settings")

    def change(self, **values):
        try:
            selected = AppSettings.model_validate(self.settings.model_dump() | values)
        except ValueError as exc:
            self.error = str(exc).splitlines()[0]
            return
        self.settings, self.error = selected, None
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

    def close(self):
        self.writer.shutdown(wait=True)

    def draw(self):
        s = self.settings
        self.theme.heading("Mesh")
        imgui.text(f"Mesh memory: {s.mesh_memory} ToF frames")
        fewer = [m for m in MEMORY_STEPS if m < s.mesh_memory]
        more = [m for m in MEMORY_STEPS if m > s.mesh_memory]
        left, right = self.layout.row(1, 1)
        if self.layout.button("Fewer", "memory fewer", left, disabled=not fewer):
            self.change(mesh_memory=fewer[-1])
        imgui.same_line()
        if self.layout.button("More", "memory more", right, disabled=not more):
            self.change(mesh_memory=more[0])
        imgui.text_disabled("Fewer: the mesh follows the latest view. More: smoother, steadier.")
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
