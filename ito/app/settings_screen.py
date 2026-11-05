import logging
from concurrent.futures import ThreadPoolExecutor

from imgui_bundle import imgui

from ito import diagnostics, recon
from ito.desktop.settings import KEYFRAMES, PRESETS, AppSettings, load, save

# Fewer / More on the settings screen
MEMORY_STEPS = (2, 3, 5, 7, 10, 15, 30, 45, 60, 75, 90, 105, 120, 240)
SHARE_STEPS = (0, 10, 25, 50, 75, 100)  # Less / More for a depth model's weights (%)
RESOLUTIONS = (25, 50, 75, 100)  # MASt3R-SLAM's input resolution (%)
# The settings screen knows no camera: map sizes are costed for a 16:9 one, as Microduck's.
COSTED = (1280, 720)


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

    def choice(self, label, name, chosen, width=None):
        """A button of a set of which one is chosen (shown pressed)."""
        if chosen:
            imgui.push_style_color(
                imgui.Col_.button, imgui.get_style_color_vec4(imgui.Col_.button_active)
            )
        pressed = self.layout.button(label, name, width)
        if chosen:
            imgui.pop_style_color()
        return pressed and not chosen

    def reconstruction(self, s):
        """The method (ito.recon.METHODS) and, where it has several, its model size."""
        self.theme.heading("Reconstruction")
        imgui.text_disabled("How the robot's surroundings are built, for a whole connection.")
        widths = self.layout.row(1, 1)  # two to a row
        for index, (value, method) in enumerate(recon.METHODS.items()):
            if index % 2:
                imgui.same_line()
            chosen = s.reconstruction == value
            if self.choice(method.label, f"method {value}", chosen, widths[index % 2]):
                self.change(reconstruction=value)
        method = recon.METHODS[s.reconstruction]
        imgui.text_wrapped(method.summary)
        size = None
        if method.sizes:
            key = f"{s.reconstruction}_size"
            size = getattr(s, key)
            imgui.text("Model size")
            widths = self.layout.row(*([1] * len(method.sizes)))
            for index, ((value, label), width) in enumerate(zip(method.sizes, widths, strict=True)):
                if index:
                    imgui.same_line()
                if self.choice(label, f"{key} {value}", size == value, width):
                    self.change(**{key: value})
        if s.reconstruction not in ("guessed", "mast3r_slam"):
            imgui.text("Precision")
            widths = self.layout.row(1, 1)
            for index, value in enumerate(("full", "half")):
                if index:
                    imgui.same_line()
                chosen = s.depth_precision == value
                if self.choice(value.title(), f"depth precision {value}", chosen, widths[index]):
                    self.change(depth_precision=value)
            imgui.text_disabled("Half: faster, less VRAM; may lose detail.")
        error = recon.missing(s.reconstruction, size)
        if error:
            imgui.text_colored(self.theme.color("error"), error)
            self.placed("model missing")

    def mesh(self, s):
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
        if s.reconstruction != "guessed":
            imgui.text("The depth model's maps, counted as a share of a ToF reading")
            self.share("Inside the ToF's view", "depth_inside", s.depth_inside)
            self.share("Outside the ToF's view", "depth_outside", s.depth_outside)
            imgui.text_disabled("Inside: measured surfaces only. Unmeasured holes still fill.")

    def slam(self, s):
        """MASt3R-SLAM's own settings (restored from main)."""
        from ito.recon.mast3r_slam.slam import map_bytes

        self.theme.heading("MASt3R-SLAM")
        imgui.text("Graphics card memory (VRAM)")
        widths = self.layout.row(*(1,) * len(PRESETS))
        for index, (label, values) in enumerate(PRESETS.items()):
            if index:
                imgui.same_line()
            chosen = all(getattr(s, key) == value for key, value in values.items())
            if self.choice(label, f"preset {label}", chosen, widths[index]):
                self.change(**values)
        imgui.text_disabled(
            "Sets the three below for a card with this much VRAM: 8 GB Half and 75%; 6 GB "
            "Half, 50% and 64 keyframes."
        )
        imgui.text("Precision")
        widths = self.layout.row(1, 1)
        for index, value in enumerate(("full", "half")):
            if index:
                imgui.same_line()
            chosen = s.slam_precision == value
            if self.choice(value.title(), f"precision {value}", chosen, widths[index]):
                self.change(slam_precision=value)
        imgui.text_disabled("Half: faster, less VRAM; may lose detail.")
        self.steps(
            "Input resolution", "slam_input_resolution", RESOLUTIONS, s.slam_input_resolution, "%"
        )
        imgui.text_disabled(
            "Lower: faster, less VRAM, blurrier. Keeps the whole frame. Below 100% leaves "
            "upstream MASt3R-SLAM's 512 pixels, which it is tuned for."
        )
        self.steps("Map size (keyframes)", "slam_keyframes", KEYFRAMES, s.slam_keyframes)
        costs = ", ".join(
            f"{count}: {map_bytes(count, *COSTED, s.slam_input_resolution) / 1e9:.1f} GB"
            for count in KEYFRAMES
        )
        imgui.text_disabled(f"VRAM at {s.slam_input_resolution}% input (16:9 camera): {costs}.")
        imgui.text_disabled(
            "Taken at start, besides MASt3R's weights and retrieval (about 3.0 GB Full, 1.7 GB "
            "Half). 512 is upstream's. A full map tracks and relocalises on but maps nothing new."
        )
        imgui.text(f"Max splats: {s.max_splats:,}")
        left, right = self.layout.row(1, 1)
        fewer = max(1, ((s.max_splats - 1) // 100_000) * 100_000)
        more = min((1 << 31) - 1, (s.max_splats // 100_000 + 1) * 100_000)
        if self.layout.button("Fewer", "splats fewer", left, disabled=s.max_splats == 1):
            self.change(max_splats=fewer)
        imgui.same_line()
        if self.layout.button("More", "splats more", right, disabled=s.max_splats == more):
            self.change(max_splats=more)
        imgui.text_disabled("More: more scene detail, more VRAM and draw time.")

    def steps(self, label, key, steps, value, unit=""):
        """A setting stepped through `steps` with - and +."""
        imgui.text(label)
        index = steps.index(value)
        minus, middle, plus = self.layout.row(1, 1.85, 1)
        if self.layout.button("-", f"{key} minus", minus, disabled=index == 0):
            self.change(**{key: steps[index - 1]})
        imgui.same_line()
        with self.theme.font("numeric"):
            self.layout.button(f"{value}{unit}", f"{key} value", middle, disabled=True)
        imgui.same_line()
        last = len(steps) - 1
        if self.layout.button("+", f"{key} plus", plus, disabled=index == last):
            self.change(**{key: steps[index + 1]})

    def share(self, label, key, value):
        imgui.text(f"{label}: {value} %")
        less = [m for m in SHARE_STEPS if m < value]
        more = [m for m in SHARE_STEPS if m > value]
        left, right = self.layout.row(1, 1)
        if self.layout.button("Less", f"{key} less", left, disabled=not less):
            self.change(**{key: less[-1]})
        imgui.same_line()
        if self.layout.button("More", f"{key} more", right, disabled=not more):
            self.change(**{key: more[0]})

    def draw(self):
        s = self.settings
        self.reconstruction(s)
        if recon.METHODS[s.reconstruction].scene == "mesh":
            self.mesh(s)
        else:
            self.slam(s)
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
