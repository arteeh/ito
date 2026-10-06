"""The same ImGui panel can render into the desktop or a headset framebuffer."""

from dataclasses import dataclass

import moderngl
import pygame
from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer


@dataclass(frozen=True)
class PilotStatus:
    link: str = "OFFLINE"
    latency_ms: float | None = None
    robot: str = "No robot connected"
    e_stop: bool = False  # Reported by the driver, never inferred from a button press.


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

    def begin(self, events, size, captured):
        for event in events:
            if event.type == pygame.VIDEORESIZE:
                continue  # SDL owns the context; resizing must not recreate it.
            if event.type == pygame.TEXTINPUT:
                self.backend.io.add_input_characters_utf8(event.text)
                continue
            if event.type == pygame.KEYDOWN:
                event.unicode = ""  # SDL TEXTINPUT also supports composed keyboard text.
            if captured and event.type in (
                pygame.MOUSEMOTION,
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP,
            ):
                continue
            self.backend.process_event(event)
        self.backend.io.display_size = size
        self.backend.process_inputs()
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
        target: moderngl.Framebuffer | None = None,
    ) -> int | None:
        if target is not None:
            target.use()
        latency = "--" if status.latency_ms is None else f"{status.latency_ms:.0f} ms"
        scene_age = "file" if age is None else f"{age:.1f} s old"
        safety = (
            "E-STOP LATCHED"
            if status.e_stop
            else "E-STOP: not latched"
            if status.link != "OFFLINE"
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
        imgui.text(f"{status.link} | latency {latency} | {status.robot}")
        imgui.text_colored((1, 0.45, 0.4, 1) if status.e_stop else (0.85, 0.9, 0.95, 1), safety)
        imgui.text(f"{count:,} splats | {fps:.0f} fps | scene {scene_age}")
        imgui.text("WASD move | PgUp/PgDn rise/fall | Home recenter")
        imgui.text("Tab release mouse" if captured else "Click scene / Tab for mouse-look")
        imgui.text("Space stop | E e-stop | R resume | F12 capture | Esc quit")
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
        if request:
            imgui.text(request)
        if self.error:
            imgui.text_colored((1, 0.45, 0.4, 1), self.error)
        imgui.end()
        imgui.render()
        self.backend.render(imgui.get_draw_data())
        return selected

    def close(self):
        self.backend.shutdown()
        imgui.destroy_context(self.imgui)
