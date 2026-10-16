"""The same ImGui panel can render into the desktop or a headset framebuffer."""

import math
from dataclasses import dataclass

import moderngl
import pygame
from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer

from ito.ui import FADE, Theme


@dataclass(frozen=True)
class PilotStatus:
    link: str = "OFFLINE"
    latency_ms: float | None = None
    robot: str = "No robot connected"
    e_stop: bool = False  # Reported by the driver, never inferred from a button press.
    robot_state: str = ""  # The driver's last reported state; "active" means it obeys input.
    detail: str = ""
    input_latency_ms: float | None = None
    # Logged rates: pilot poses, joined camera frames, backend calls (including pose-only).
    rates: tuple[float | None, float | None, float | None] = (None, None, None)
    motion_hz: float | None = None  # The robot's own camera motion; None: it sends none.
    reconstruction: str = ""
    audio: str = ""
    robot_audio: str = ""
    mic_muted: bool = False
    speaker_muted: bool = False
    robot_microphone: bool = False
    robot_speaker: bool = False
    armed: bool = False  # Movement reaches the robot only after an explicit resume.
    link_lagging: bool = False
    focus_hold: bool = False  # Stopped only because the window lost focus.


class Overlay:
    def __init__(self, context: moderngl.Context, max_splats: int):
        self.context = context
        self.imgui = imgui.create_context()
        imgui.get_io().set_ini_filename("")
        self.theme = Theme()
        self.backend = PygameRenderer()
        self.backend.key_map.update(
            {getattr(pygame, f"K_{key}"): getattr(imgui.Key, key) for key in "acvxyz"}
        )
        self.max_splats = max_splats
        self.error = None
        self.commands = []
        self._leave = False
        self.can_leave = self.leave = False  # Disconnect returns to the connect screen.
        self.simulation = None
        self.captured = False
        self.layout = {}  # Control centres, for scripted clicks (ITO_PILOT_SCRIPT).
        self.prompt_status = None
        self.nudged = -math.inf  # When the pilot last tried to move a robot that is stopped.

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
            # The backend hands any button to ImGui, whose assert on a slot it lacks kills
            # the display: the legacy wheel buttons (4, 5) and side buttons (6, 7) go no further.
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and event.button > 3:
                continue
            # Tab toggles mouse-look. Outside a text field, ImGui would take it for keyboard
            # navigation and from then on claim the keyboard, so Tab could no longer grab.
            # Its release still arrives, like mouse releases above.
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_TAB
                and not io.want_text_input
            ):
                continue
            self.backend.process_event(event)
        self.backend.io.display_size = size
        self.backend.process_inputs()
        if pointer is not None:
            x, y, down = pointer
            self.backend.io.add_mouse_pos_event(x, y)
            self.backend.io.add_mouse_button_event(0, down)
        imgui.new_frame()
        self.theme.begin(size)
        return imgui.get_io()

    @property
    def leave(self):
        return self._leave and imgui.get_time() - self.theme.switched >= FADE

    @leave.setter
    def leave(self, requested):
        self._leave = requested
        if requested:
            self.theme.leave()
        elif self.theme.leaving:
            self.theme.name = self.theme.previous = None
            self.theme.leaving = False
            self.theme.start_leave = False

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
        performance: dict | None = None,
        target: moderngl.Framebuffer | None = None,
        xr_mode=False,
        driving=False,
        panel_shown=True,
    ) -> int | None:
        if target is not None:
            target.use()
        if not self._leave:
            self.theme.show("hud", "session")
        _, alpha, leaving = self.theme.showing()
        hud_alpha = self.theme.hud.show(panel_shown, self.theme.t)
        selected = None
        with self.theme.faded(alpha, leaving):
            if hud_alpha > 0:
                with self.theme.faded(hud_alpha, not panel_shown):
                    selected = self._draw_hud(
                        status,
                        fps,
                        count,
                        age,
                        captured,
                        request,
                        live=live,
                        performance=performance,
                        xr_mode=xr_mode,
                        driving=driving,
                    )
            self._stopped_prompt(status, driving, xr_mode)
        self.render()
        return selected

    def _draw_hud(
        self,
        status: PilotStatus,
        fps: float,
        count: int,
        age: float | None,
        captured: bool,
        request: str | None,
        *,
        live=False,
        performance: dict | None = None,
        xr_mode: bool = False,
        driving: bool = False,
    ) -> int | None:
        safety = (
            "E-STOP LATCHED"
            if status.e_stop
            else "E-STOP: not latched"
            if status.link == "CONNECTED"
            else "E-STOP: unavailable offline"
        )
        if xr_mode:
            # The VR quad hangs centred ahead of the pilot: centre the window on it. A steady
            # minimum width keeps the controls from shifting as status text changes length.
            width, height = imgui.get_io().display_size
            maximum_height = height - 36
            stopped = status.link == "CONNECTED" and (
                not status.armed or status.e_stop or status.detail.startswith("fault")
            )
            if stopped or self.theme.stopped.alpha(self.theme.t) > 0:
                # Reserve the safety prompt's two display lines and body line. Long status
                # details scroll within the HUD instead of hiding a safety control.
                style = imgui.get_style()
                scale = style.font_scale_main * style.font_scale_dpi
                prompt_height = (
                    imgui.get_font_size() * (3.2 * scale + 1)
                    + 3 * style.item_spacing.y
                    + 2 * style.window_padding.y
                )
                maximum_height -= prompt_height + 12
            imgui.set_next_window_pos((width / 2, 12), imgui.Cond_.always, (0.5, 0))
            imgui.set_next_window_size_constraints(
                (min(580, width - 24), 0), (width - 24, max(100, maximum_height))
            )
        else:
            imgui.set_next_window_pos((12, 12), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(0.9)
        with self.theme.font("title"):
            imgui.begin(
                "Ito",
                flags=imgui.WindowFlags_.always_auto_resize
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_collapse,
            )
        self.theme.window("Ito", "hud", "under")
        latency = "--" if status.latency_ms is None else f"{status.latency_ms:.0f} ms"
        with self.theme.font("numeric"):
            imgui.text(f"{status.link} | RTT {latency} | {fps:.0f} fps | {status.robot}")
        imgui.text_colored(
            self.theme.color("error") if status.e_stop else self.theme.color("safe"), safety
        )
        if not live:
            scene_age = "file" if age is None else f"{age:.1f} s old"
            with self.theme.font("numeric"):
                imgui.text(f"{count:,} splats | scene {scene_age}")
        if live:
            values = performance or {}
            for label, key, unit in (
                ("Reconstruction", "scene_hz", "frames/s"),
                ("Capture to visible", "capture_to_visible_ms", "ms"),
                ("GPU busy", "gpu_busy_pct", "%"),
            ):
                value = values.get(key)
                text = "--" if value is None else f"{value:.1f}"
                with self.theme.font("numeric"):
                    imgui.text(f"{label}: {text} {unit}")
            # Splats in the scene of the limit now: Max splats, or less if VRAM is short.
            used, limit = values.get("slot_count"), values.get("splat_limit")
            splats = "--" if used is None or limit is None else f"{used:,} / {limit:,}"
            vram = values.get("vram")
            memory = "--" if vram is None else f"{vram[0] / 2**30:.1f} / {vram[1] / 2**30:.1f} GiB"
            with self.theme.font("numeric"):
                imgui.text(f"Splats: {splats} | VRAM {memory}")
            if values.get("slam_precision"):
                imgui.text(f"MASt3R: {values['slam_precision']}")
            if values.get("reconstruction_oom"):
                imgui.text_colored(
                    self.theme.color("error"), "Out of GPU memory: reconstruction is degraded"
                )
            if values.get("network_nonfinite_points"):
                imgui.text_colored(
                    self.theme.color("warning"), "fp16 overflow: invalid points discarded"
                )
        if xr_mode:
            imgui.text("Aim + trigger to select | stick to drive")
            imgui.text("B e-stop | X stop | A resume | Y recenter")
        else:
            imgui.text("WASD move | PgUp/PgDn rise/fall | Home recenter")
            imgui.text("Tab release mouse" if captured else "Click scene / Tab for mouse-look")
            imgui.text("Space stop | E e-stop | R resume | F12 capture | Esc release mouse")
        selected = None
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
                imgui.text_colored(self.theme.color("error"), self.simulation.viewer_error)
        if request:
            imgui.text(request)
        if status.detail:
            imgui.text(status.detail)
        if status.reconstruction and "tracking |" not in status.reconstruction:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 550)
            imgui.text_wrapped(status.reconstruction)
            imgui.pop_text_wrap_pos()
        if self.error:
            imgui.text_colored(self.theme.color("error"), self.error)
        link = "cut" if status.link != "CONNECTED" else "slack" if status.link_lagging else "taut"
        self.theme.window("Ito", "hud", "over", link=link)
        imgui.end()
        return selected

    def _stopped_prompt(self, status, driving, xr_mode):
        """Movement is ignored until resume; the pilot must never have to guess why."""
        fault = status.detail.startswith("fault")
        visible = status.link == "CONNECTED" and (not status.armed or status.e_stop or fault)
        alpha = self.theme.stopped.show(visible, self.theme.t)
        if visible:
            self.prompt_status = status
        if alpha <= 0 or self.prompt_status is None:
            return
        with self.theme.faded(alpha, not visible):
            self._draw_prompt(self.prompt_status, driving and visible, xr_mode)

    def _draw_prompt(self, status, driving, xr_mode):
        fault = status.detail.startswith("fault")
        now = self.theme.t
        if driving:
            self.nudged = now
        resume = "Press A" if xr_mode else "Press R (or gamepad A)"
        if fault:
            title = "Robot fault: " + status.detail.removeprefix("fault: ")
            action = "Resume cannot clear a fault; restart the robot driver"
        elif status.e_stop:
            title, action = "E-STOP LATCHED", f"{resume} to resume"
        elif status.focus_hold and not xr_mode:
            title = "Robot stopped: window lost focus"
            action = "Click the scene or press R (or gamepad A) to drive"
        else:
            title, action = "Robot stopped", f"{resume} to drive"
        nudged = now - self.nudged < 1.5
        # 2 Hz stays below photosensitivity limits; the headset gets a steady highlight,
        # since flashing in the pilot's view costs comfort.
        bright = nudged and (xr_mode or math.sin((now - self.nudged) * 4 * math.pi) > -0.3)
        role = "estop_bg" if status.e_stop or fault else "stop_bg"
        background = self.theme.color(role + ("_bright" if bright else ""))
        width, height = imgui.get_io().display_size
        imgui.set_next_window_pos((width / 2, height - 24), imgui.Cond_.always, (0.5, 1))
        imgui.push_style_color(imgui.Col_.window_bg, background)
        imgui.push_style_color(
            imgui.Col_.border, self.theme.color("prompt_border") if nudged else (0, 0, 0, 0)
        )
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 3)
        imgui.begin(
            "Robot stopped",
            flags=imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.always_auto_resize
            | imgui.WindowFlags_.no_inputs
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_nav,
        )
        self.theme.window("Robot stopped", "prompt", "under")
        with self.theme.font("heading", 1.6):
            imgui.text(title)
            imgui.text(action)
        imgui.text("Movement is ignored until you resume.")
        self.theme.window("Robot stopped", "prompt", "over")
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color(2)

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
