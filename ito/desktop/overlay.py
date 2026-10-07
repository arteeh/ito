"""The same ImGui panel can render into the desktop or a headset framebuffer."""

from dataclasses import dataclass

import moderngl
import pygame
from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer

from ito import diagnostics


@dataclass(frozen=True)
class PilotStatus:
    link: str = "OFFLINE"
    latency_ms: float | None = None
    robot: str = "No robot connected"
    e_stop: bool = False  # Reported by the driver, never inferred from a button press.
    detail: str = ""
    input_latency_ms: float | None = None
    reconstruction: str = ""
    audio: str = ""
    robot_audio: str = ""
    mic_muted: bool = False
    speaker_muted: bool = False
    robot_microphone: bool = False
    robot_speaker: bool = False


class Overlay:
    def __init__(self, context: moderngl.Context, max_splats: int):
        self.context = context
        self.imgui = imgui.create_context()
        imgui.get_io().set_ini_filename("")
        self.backend = PygameRenderer()
        self.backend.key_map.update(
            {getattr(pygame, f"K_{key}"): getattr(imgui.Key, key) for key in "acvxyz"}
        )
        self.max_splats = max_splats
        self.error = None
        self.commands = []
        self.can_leave = self.leave = False  # Disconnect returns to the connect screen.
        self.simulation = None
        self.captured = False
        self.layout = {}  # Where the last frame drew each control, so e2e clicks what pilots see.

    def begin(self, events, size, captured, *, pointer=None):
        io = self.backend.io
        if captured != self.captured:
            # The hidden cursor must not hover controls; on release it is back where SDL left it.
            io.add_mouse_pos_event(*((-imgui.FLT_MAX,) * 2 if captured else pygame.mouse.get_pos()))
            self.captured = captured
        for event in events:
            if event.type == pygame.VIDEORESIZE:
                continue  # SDL owns the context; resizing must not recreate it.
            if event.type == pygame.TEXTINPUT:
                self.backend.io.add_input_characters_utf8(event.text)
                continue
            if event.type == pygame.KEYDOWN:
                event.unicode = ""  # SDL TEXTINPUT also supports composed keyboard text.
            # Releases still reach ImGui: the click that starts mouse-look is released while
            # captured, and a button ImGui believes held swallows every later click.
            if captured and event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                continue
            self.backend.process_event(event)
        self.backend.io.display_size = size
        self.backend.process_inputs()
        if pointer is not None:
            x, y, down = pointer
            self.backend.io.add_mouse_pos_event(x, y)
            self.backend.io.add_mouse_button_event(0, down)
        imgui.new_frame()
        return imgui.get_io()

    def draw(
        self,
        status: PilotStatus,
        fps: float,
        count: int,
        age: float | None,
        captured: bool,
        request: str | None,
        *,
        live=False,
        capture_latency_ms: float | None = None,
        target: moderngl.Framebuffer | None = None,
        xr_mode: bool = False,
    ) -> int | None:
        if target is not None:
            target.use()
        latency = "--" if status.latency_ms is None else f"{status.latency_ms:.0f} ms"
        scene_age = "file" if age is None else f"{age:.1f} s old"
        safety = (
            "E-STOP LATCHED"
            if status.e_stop
            else "E-STOP: not latched"
            if status.link == "CONNECTED"
            else "E-STOP: unavailable offline"
        )
        imgui.set_next_window_pos((12, 12), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(0.9)
        imgui.begin(
            "Ito",
            flags=imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_collapse,
        )
        imgui.text(f"{status.link} | RTT {latency} | {status.robot}")
        imgui.text_colored((1, 0.45, 0.4, 1) if status.e_stop else (0.85, 0.9, 0.95, 1), safety)
        imgui.text(f"{count:,} splats | {fps:.0f} fps | scene {scene_age}")
        if xr_mode:
            imgui.text("Aim + trigger to select | stick to drive")
            imgui.text("B e-stop | X stop | A resume | Y recenter")
        else:
            imgui.text("WASD move | PgUp/PgDn rise/fall | Home recenter")
            imgui.text("Tab release mouse" if captured else "Click scene / Tab for mouse-look")
            imgui.text("Space stop | E e-stop | R resume | F12 capture | Esc release mouse")
        selected = None
        if live:
            imgui.set_next_item_width(160)
            _, self.max_splats = imgui.input_int("Max splats", self.max_splats, 1024, 16384)
            imgui.same_line()
            if imgui.button("Apply"):
                if 1 <= self.max_splats <= 4_194_304:
                    selected = self.max_splats
                    self.error = None
                else:
                    self.error = "Choose between 1 and 4,194,304 splats"
        controls = [("Stop", "stop"), ("E-stop", "e_stop"), ("Resume", "resume")]
        if xr_mode:
            controls.append(("Recenter", "recenter"))
        for index, (label, command) in enumerate(controls):
            if index:
                imgui.same_line()
            if imgui.button(label):
                self.commands.append(command)
            self._placed(command)
        if self.can_leave:
            imgui.same_line()
            if imgui.button("Disconnect"):
                self.leave = True
            self._placed("disconnect")
        if status.link == "CONNECTED":
            # Above the status lines that come and go, so the toggles never move under a click.
            self._audio(status)
        if self.simulation:
            changed, visible = imgui.checkbox("Show simulation", self.simulation.visible)
            self._placed("show_simulation")
            if changed:
                try:
                    self.simulation.show(visible)
                except OSError:
                    self.error = "Could not launch simulation viewer"
            if self.simulation.viewer_error:
                imgui.text_colored((1, 0.45, 0.4, 1), self.simulation.viewer_error)
        self.diagnostic_controls()
        if request:
            imgui.text(request)
        if status.detail:
            imgui.text(status.detail)
        if status.reconstruction:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 550)
            imgui.text_wrapped(status.reconstruction)
            imgui.pop_text_wrap_pos()
        if status.input_latency_ms is not None:
            imgui.text(f"Pilot input -> robot: {status.input_latency_ms:.1f} ms")
        if capture_latency_ms is not None:
            imgui.text(f"Camera capture -> splat visible: {capture_latency_ms:.1f} ms")
        if self.error:
            imgui.text_colored((1, 0.45, 0.4, 1), self.error)
        imgui.end()
        self.render()
        return selected

    def diagnostic_controls(self):
        debug = diagnostics.current()
        imgui.begin_disabled(debug.override is not None)
        changed, enabled = imgui.checkbox("Diagnostic logging", debug.enabled)
        self._placed("diagnostics")
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        imgui.end_disabled()
        if changed:
            debug.set_enabled(enabled)
        if debug.override is not None:
            imgui.same_line()
            imgui.text_disabled("ITO_DEBUG override")
        if debug.error:
            imgui.text_colored((1, 0.45, 0.4, 1), debug.error)
        elif hovered:
            imgui.set_tooltip(str(debug.path))

    def _audio(self, status):
        missing = [
            name
            for name, present in (
                ("microphone", status.robot_microphone),
                ("speaker", status.robot_speaker),
            )
            if not present
        ]
        if missing:
            imgui.text_disabled("Robot has no " + " or ".join(missing))
        if len(missing) == 2:
            return
        imgui.text(status.audio)
        controls = []
        if status.robot_speaker:
            controls.append(("Mute microphone", status.mic_muted, "mute_mic"))
        if status.robot_microphone:
            controls.append(("Mute speakers", status.speaker_muted, "mute_speaker"))
        for index, (label, muted, command) in enumerate(controls):
            if index:
                imgui.same_line()
            if imgui.checkbox(label, muted)[0]:
                self.commands.append(command)
            self._placed(command)
        if status.robot_audio:
            imgui.text("Robot " + status.robot_audio)

    def _placed(self, name):
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        self.layout[name] = ((low.x + high.x) / 2, (low.y + high.y) / 2)

    def render(self):
        imgui.render()
        self.backend.render(imgui.get_draw_data())

    def close(self):
        self.backend.shutdown()
        imgui.destroy_context(self.imgui)
