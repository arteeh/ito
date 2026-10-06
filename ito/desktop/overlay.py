from dataclasses import dataclass

import moderngl
import pygame


@dataclass(frozen=True)
class PilotStatus:
    link: str = "OFFLINE"
    latency_ms: float | None = None
    robot: str = "No robot connected"
    e_stop: bool = False  # Reported by the driver, never inferred from a button press.


class Overlay:
    def __init__(self, context: moderngl.Context):
        self.context = context
        self.font = pygame.font.Font(None, 23)
        self.program = context.program(vertex_shader='''#version 430
            out vec2 uv;
            uniform vec2 size;
            uniform vec2 viewport;
            void main() {
                uv = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);
                vec2 pixel = vec2(12) + uv * size;
                gl_Position = vec4(pixel / viewport * vec2(2,-2) + vec2(-1,1), 0, 1);
            }''', fragment_shader='''#version 430
            in vec2 uv;
            uniform sampler2D panel;
            out vec4 color;
            void main() { color = texture(panel, uv); }''')
        self.program["panel"] = 1
        self.vao = context.vertex_array(self.program, [])
        self.texture = None
        self.previous = None

    def draw(self, status: PilotStatus, fps: float, count: int, age: float | None,
             captured: bool, request: str | None) -> None:
        latency = "--" if status.latency_ms is None else f"{status.latency_ms:.0f} ms"
        scene_age = "file" if age is None else f"{age:.1f} s old"
        safety = "E-STOP LATCHED" if status.e_stop else "E-STOP: not latched" if status.link != "OFFLINE" else "E-STOP: unavailable offline"
        lines = (
            f"ITO   {status.link}   |   latency {latency}",
            f"Robot: {status.robot}   |   {safety}",
            f"{count:,} splats   |   {fps:.0f} fps   |   scene {scene_age}",
            "WASD move  PgUp/PgDn rise/fall  Home recenter",
            f"{'Tab release mouse' if captured else 'Click / Tab mouse-look'}  |  Gamepad: sticks move/look, shoulders rise/fall",
            "Space / X stop   E / B e-stop   R / A resume   F12 capture   Esc quit",
            request or "",
        )
        if lines != self.previous:
            color = (255, 120, 115) if status.e_stop else (220, 231, 240)
            rendered = [self.font.render(line, True, color if i == 1 else (220, 231, 240))
                        for i, line in enumerate(lines)]
            size = (max(text.get_width() for text in rendered) + 24, len(lines) * 24 + 16)
            panel = pygame.Surface(size, pygame.SRCALPHA)
            panel.fill((13, 20, 31, 226))
            for i, text in enumerate(rendered):
                panel.blit(text, (12, 8 + i * 24))
            if self.texture is not None:
                self.texture.release()
            self.texture = self.context.texture(size, 4, pygame.image.tobytes(panel, "RGBA"))
            self.texture.filter = moderngl.LINEAR, moderngl.LINEAR
            self.previous = lines
        self.context.enable_only(moderngl.BLEND)
        self.context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.texture.use(1)
        self.program["size"] = self.texture.size
        self.program["viewport"] = self.context.viewport[2:]
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4)

    def close(self) -> None:
        if self.texture is not None:
            self.texture.release()
        self.vao.release()
        self.program.release()
