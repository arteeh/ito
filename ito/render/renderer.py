"""OpenGL 4.3 renderer. No CUDA, CPU sorting, or GPU readback in the draw path."""

import time
from importlib.resources import files

import moderngl
import numpy as np

from .pose import validate_pose
from .scene import FloatArray, GaussianBuffer


class GaussianRenderer:
    def __init__(self, context: moderngl.Context):
        if context.version_code < 430:
            raise RuntimeError("Ito needs OpenGL 4.3 compute shaders (Mesa llvmpipe is supported)")
        self.context = context
        shaders = files("ito.render").joinpath("shaders")
        self.keys = context.compute_shader(shaders.joinpath("keys.glsl").read_text())
        self.sort = context.compute_shader(shaders.joinpath("sort.glsl").read_text())
        self.program = context.program(
            vertex_shader=shaders.joinpath("splat.vert").read_text(),
            fragment_shader=shaders.joinpath("splat.frag").read_text(),
        )
        self.vao = context.vertex_array(self.program, [])
        self.update = context.compute_shader(shaders.joinpath("update.glsl").read_text())
        self.changes = context.buffer(reserve=4096 * 64)
        self.slots = context.buffer(reserve=4096 * 4)
        self.epoch = 0.0
        self.fade_seconds = 0.0
        self.uploaded_bytes = 0
        self.draw_count = 0
        self.scene_buffer = None
        self.order = None
        self.count = 0
        self.capacity = 0
        self.stride = 4
        self.degree = 0

    def upload(self, scene: GaussianBuffer) -> None:
        """Consume a stable published snapshot. The producer may reuse it after return."""
        count, stride, _ = scene.records.shape
        capacity = max(256, 1 << max(0, count - 1).bit_length())
        limit = self.context.info["GL_MAX_SHADER_STORAGE_BLOCK_SIZE"]
        if max(count * stride * 16, capacity * 8) > limit:
            raise ValueError(f"Scene with {count:,} Gaussians exceeds OpenGL buffer limits")
        buffer = self.context.buffer(reserve=max(64, count * stride * 16))
        if count:
            buffer.write(scene.records)
        try:
            order = self.context.buffer(reserve=capacity * 8)
        except Exception:
            buffer.release()
            raise
        if self.scene_buffer is not None:
            self.scene_buffer.release()
            self.order.release()
        self.scene_buffer, self.order = buffer, order
        self.count, self.capacity, self.stride = count, capacity, stride
        self.degree = scene.sh_degree
        self.draw_count = count
        self.fade_seconds = 0.0
        self.uploaded_bytes += scene.records.nbytes

    def apply(self, update) -> None:
        """Scatter a bounded incremental packet on GPU; never read back the scene."""
        required = update.capacity
        new_scene = self.epoch != update.epoch
        if self.fade_seconds == 0 or new_scene or required > self.draw_count:
            capacity = max(256, 1 << max(0, required - 1).bit_length())
            limit = self.context.info["GL_MAX_SHADER_STORAGE_BLOCK_SIZE"]
            if max(required * 64, capacity * 8) > limit:
                raise ValueError("Splat budget exceeds OpenGL buffer limits")
            scene = self.context.buffer(reserve=required * 64)
            scene.clear()
            order = self.context.buffer(reserve=capacity * 8)
            if self.scene_buffer is not None:
                if self.fade_seconds and not new_scene:
                    self.context.copy_buffer(scene, self.scene_buffer, size=self.draw_count * 64)
                self.scene_buffer.release()
                self.order.release()
            self.scene_buffer, self.order = scene, order
            self.capacity, self.draw_count = capacity, required
        self.count = update.count
        self.stride, self.degree = 4, 0
        self.epoch, self.fade_seconds = update.epoch, update.fade_seconds
        n = len(update.indices)
        if not n:
            return
        # Orphan staging storage so queued GPU reads never fence the next CPU upload.
        self.changes.orphan(max(self.changes.size, n * 64))
        self.slots.orphan(max(self.slots.size, n * 4))
        records = update.records.copy()
        retiring = records[:, 3, 3] == 1
        records[retiring, 1, 3] = np.minimum(
            records[retiring, 1, 3], time.monotonic() - self.epoch + self.fade_seconds
        )
        self.changes.write(records)
        self.slots.write(update.indices)
        self.uploaded_bytes += update.records.nbytes + update.indices.nbytes
        self.scene_buffer.bind_to_storage_buffer(1)
        self.changes.bind_to_storage_buffer(2)
        self.slots.bind_to_storage_buffer(3)
        self.update["count"] = n
        self.update.run(group_x=(n + 255) // 256)
        self.context.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        if update.acknowledge is not None:
            update.acknowledge()

    def draw(
        self,
        robot_camera: FloatArray,
        head: FloatArray,
        projection: FloatArray,
        target: moderngl.Framebuffer | None = None,
        *,
        clear: tuple[float, float, float, float] | None = (0.025, 0.035, 0.055, 1.0),
        viewport: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Draw world-space splats from world_from_camera @ camera_from_head.

        Call once per eye with its current pose, projection and framebuffer. Scene
        updates and anchor updates are independent; a stalled producer never stalls
        tracking. Projection uses OpenGL's [-1, 1] clip depth. Owns GL draw state.
        """
        world_from_eye = validate_pose(robot_camera) @ validate_pose(head)
        projection = np.asarray(projection, dtype=np.float32)
        if projection.shape != (4, 4) or not np.isfinite(projection).all():
            raise ValueError("Projection must be a finite 4x4 matrix")
        target = target if target is not None else self.context.screen
        if target is None:
            raise ValueError("An offscreen context needs an explicit framebuffer")
        target.use()
        viewport = viewport if viewport is not None else (0, 0, *target.size)
        if min(viewport[2:]) <= 0:
            raise ValueError("Viewport dimensions must be positive")
        self.context.viewport = viewport
        self.context.scissor = None
        self.context.enable_only(moderngl.BLEND)
        self.context.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.context.blend_equation = moderngl.FUNC_ADD
        if clear is not None:
            target.clear(*clear, viewport=viewport)
        if not self.draw_count:
            return
        rotation = world_from_eye[:3, :3]
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation.T
        view[:3, 3] = -rotation.T @ world_from_eye[:3, 3]
        self.scene_buffer.bind_to_storage_buffer(1)
        self.order.bind_to_storage_buffer(0)
        self.keys["view"].write(view.T.copy())
        self.keys["count"] = self.draw_count
        self.keys["capacity"] = self.capacity
        self.keys["stride"] = self.stride
        self.keys.run(group_x=self.capacity // 256)
        self.context.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        self.sort["capacity"] = self.capacity
        stage = 2
        while stage <= self.capacity:
            self.sort["stage"] = stage
            distance = stage // 2
            while distance:
                local = distance < 256
                self.sort["distance"] = distance
                self.sort["local_merge"] = local
                self.sort.run(group_x=self.capacity // 256)
                self.context.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
                distance = 0 if local else distance // 2
            stage *= 2
        self.program["view"].write(view.T.copy())
        self.program["projection"].write(projection.T.copy())
        self.program["eye"] = tuple(world_from_eye[:3, 3])
        self.program["viewport"] = viewport[2:]
        self.program["scene_time"] = time.monotonic() - self.epoch if self.fade_seconds else 0
        self.program["fade_seconds"] = self.fade_seconds
        self.program["stride"] = self.stride
        self.program["sh_degree"] = self.degree
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4, instances=self.draw_count)

    def close(self) -> None:
        for resource in (
            self.vao,
            self.program,
            self.sort,
            self.keys,
            self.scene_buffer,
            self.order,
            self.update,
            self.changes,
            self.slots,
        ):
            if resource is not None:
                resource.release()
