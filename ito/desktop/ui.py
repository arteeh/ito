"""ImGui on pygame and Vulkan: pygame's events to ImGui (PygameInput, as imgui-bundle's pygame
backend has it) and ImGui's draw data drawn with Vulkan (UiRenderer: one pipeline, the textures
ImGui asks for, its font atlas among them)."""

import ctypes

import pygame
import vulkan as vk
from imgui_bundle import imgui

from ito.render.vk import Image, Pipeline, pack

VERTEX = ((0, 0, "2f"), (1, 0, "2f"), (2, 0, "4u8"))  # pos, uv, col of ImDrawVert (20 bytes)


class PygameInput:
    def __init__(self):
        self.io = imgui.get_io()
        self.io.delta_time = 1 / 60
        self.time = None
        self.key_map = {
            pygame.K_LEFT: imgui.Key.left_arrow,
            pygame.K_RIGHT: imgui.Key.right_arrow,
            pygame.K_UP: imgui.Key.up_arrow,
            pygame.K_DOWN: imgui.Key.down_arrow,
            pygame.K_PAGEUP: imgui.Key.page_up,
            pygame.K_PAGEDOWN: imgui.Key.page_down,
            pygame.K_HOME: imgui.Key.home,
            pygame.K_END: imgui.Key.end,
            pygame.K_INSERT: imgui.Key.insert,
            pygame.K_DELETE: imgui.Key.delete,
            pygame.K_BACKSPACE: imgui.Key.backspace,
            pygame.K_RETURN: imgui.Key.enter,
            pygame.K_ESCAPE: imgui.Key.escape,
            pygame.K_KP_ENTER: imgui.Key.keypad_enter,
            pygame.K_TAB: imgui.Key.tab,
            pygame.K_LCTRL: imgui.Key.left_ctrl,
            pygame.K_RCTRL: imgui.Key.right_ctrl,
            pygame.K_LALT: imgui.Key.left_alt,
            pygame.K_RALT: imgui.Key.right_alt,
            pygame.K_RSHIFT: imgui.Key.right_shift,
            pygame.K_LSHIFT: imgui.Key.left_shift,
            pygame.K_LSUPER: imgui.Key.left_super,
            pygame.K_RSUPER: imgui.Key.right_super,
        }
        self.modifier_map = {
            pygame.K_LCTRL: imgui.Key.mod_ctrl,
            pygame.K_RCTRL: imgui.Key.mod_ctrl,
            pygame.K_LSHIFT: imgui.Key.mod_shift,
            pygame.K_RSHIFT: imgui.Key.mod_shift,
            pygame.K_LALT: imgui.Key.mod_alt,
            pygame.K_RALT: imgui.Key.mod_alt,
            pygame.K_LSUPER: imgui.Key.mod_super,
            pygame.K_RSUPER: imgui.Key.mod_super,
        }
        self.mouse_map = {2: imgui.MouseButton_.middle, 3: imgui.MouseButton_.right}

    def process_event(self, event):
        io = self.io
        if event.type == pygame.MOUSEMOTION:
            io.add_mouse_pos_event(*event.pos)
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            button = self.mouse_map.get(event.button, event.button - 1)
            io.add_mouse_button_event(button, event.type == pygame.MOUSEBUTTONDOWN)
        elif event.type == pygame.MOUSEWHEEL:
            io.add_mouse_wheel_event(event.x * 0.5, event.y * 0.5)
        elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
            down = event.type == pygame.KEYDOWN
            special = False
            for mapping in (self.key_map, self.modifier_map):
                if event.key in mapping:
                    io.add_key_event(mapping[event.key], down=down)
                    special = True
            if down and not special:
                for char in event.unicode:
                    if 0 < ord(char) < 0x10000:
                        io.add_input_character(ord(char))

    def process_inputs(self):
        now = pygame.time.get_ticks() / 1000
        self.io.delta_time = max(now - self.time, 1e-3) if self.time is not None else 1 / 60
        self.time = now


class UiRenderer:
    """Draws ImGui's draw data into a display image (Vulkan's orientation, row 0 at the top)."""

    def __init__(self, device):
        self.device = device
        self.textures = {}
        self.next_id = 1
        self.pipelines = {}
        io = imgui.get_io()
        io.backend_flags |= (
            imgui.BackendFlags_.renderer_has_textures.value
            | imgui.BackendFlags_.renderer_has_vtx_offset.value
        )
        side = device.properties.limits.maxImageDimension2D
        platform = imgui.get_platform_io()
        platform.renderer_texture_max_width = platform.renderer_texture_max_height = side

    def _pipeline(self, format):
        if format not in self.pipelines:
            self.pipelines[format] = Pipeline(
                self.device,
                ("imgui.vert", "imgui.frag"),
                {0: "sampler"},
                push=16,
                vertex=VERTEX,
                strides={0: (imgui.VERTEX_SIZE, {0: 0, 1: 8, 2: 16})},
                colours=(format,),
                blend="over",
            )
        return self.pipelines[format]

    def _textures(self, frame):
        for texture in imgui.get_platform_io().textures:
            status = texture.status
            if status == imgui.ImTextureStatus.want_create:
                size = (texture.width, texture.height)
                image = Image(self.device, size, vk.VK_FORMAT_R8G8B8A8_UNORM, sampled=True)
                frame.upload_image(image, texture.get_pixels_array(), (0, 0, *size))
                self.textures[self.next_id] = image
                texture.set_tex_id(self.next_id)
                self.next_id += 1
                texture.status = imgui.ImTextureStatus.ok
            elif status == imgui.ImTextureStatus.want_updates:
                image = self.textures[texture.tex_id]
                pixels = texture.get_pixels_array().reshape(texture.height, texture.width, 4)
                for r in texture.updates:
                    rows = pixels[r.y : r.y + r.h, r.x : r.x + r.w]
                    frame.upload_image(image, rows, (r.x, r.y, r.w, r.h))
                texture.status = imgui.ImTextureStatus.ok
            elif status == imgui.ImTextureStatus.want_destroy:
                image = self.textures.pop(texture.tex_id, None)
                if image is not None:
                    self.device.defer(image)
                texture.set_tex_id(0)
                texture.status = imgui.ImTextureStatus.destroyed
        frame.sampled(*self.textures.values())

    def render(self, frame, target):
        """ImGui's last rendered frame (imgui.render()) over target."""
        data = imgui.get_draw_data()
        self._textures(frame)
        width, height = data.display_size
        if width <= 0 or height <= 0 or not data.cmd_lists:
            return
        sx, sy = target.size[0] / width, target.size[1] / height
        x0, y0 = data.display_pos
        pipeline = self._pipeline(target.format)
        push = pack("4f", 2 / width, 2 / height, -1 - x0 * 2 / width, -1 - y0 * 2 / height)
        sampler = self.device.sampler()
        lists = []
        for commands in data.cmd_lists:
            vertices = ctypes.string_at(
                commands.vtx_buffer.data_address(), commands.vtx_buffer.size() * imgui.VERTEX_SIZE
            )
            indices = ctypes.string_at(
                commands.idx_buffer.data_address(), commands.idx_buffer.size() * imgui.INDEX_SIZE
            )
            lists.append((frame.stage(vertices), frame.stage(indices), commands))
        with frame.render([target]):
            bound = None
            for (vertices, start, _), (indices, offset, _), commands in lists:
                frame.vertices((vertices, start), index=(indices, offset, imgui.INDEX_SIZE == 2))
                for command in commands.cmd_buffer:
                    x1, y1, x2, y2 = command.clip_rect
                    x1, y1 = max((x1 - x0) * sx, 0), max((y1 - y0) * sy, 0)
                    x2 = min((x2 - x0) * sx, target.size[0])
                    y2 = min((y2 - y0) * sy, target.size[1])
                    if x2 <= x1 or y2 <= y1 or not command.elem_count:
                        continue
                    texture = command.tex_ref.get_tex_id()
                    if texture != bound:
                        frame.use(pipeline, {0: (self.textures[texture], sampler)}, push)
                        bound = texture
                    frame.area((0, 0, *target.size), (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                    frame.draw(
                        command.elem_count,
                        indexed=True,
                        first=command.idx_offset,
                        vertex_offset=command.vtx_offset,
                    )

    def close(self, *, again=False):
        """Release the GPU's side. again: a renderer on another device takes over, so ImGui
        uploads its textures afresh."""
        for texture in imgui.get_platform_io().textures:
            if texture.tex_id in self.textures:
                texture.set_tex_id(0)
                texture.status = (
                    imgui.ImTextureStatus.want_create if again else imgui.ImTextureStatus.destroyed
                )
        for image in self.textures.values():
            image.close()
        self.textures = {}
        for pipeline in self.pipelines.values():
            pipeline.close()
        self.pipelines = {}
