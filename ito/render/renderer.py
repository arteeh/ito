"""OpenGL 4.3 renderer. No CUDA, CPU sorting, or GPU readback in the draw path."""

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
        self.texture = None
        self.order = None
        self.count = 0
        self.capacity = 0
        self.stride = 4
        self.degree = 0
        self.keys["gaussians"] = 0
        self.program["gaussians"] = 0

    def upload(self, scene: GaussianBuffer) -> None:
        """Consume a stable published snapshot. The producer may reuse it after return."""
        count, stride, _ = scene.records.shape
        capacity = max(256, 1 << max(0, count - 1).bit_length())
        limit = self.context.info["GL_MAX_TEXTURE_SIZE"]
        width = min(4096, limit)
        height = max(1, (count * stride + width - 1) // width)
        if height > limit or capacity * 8 > self.context.info["GL_MAX_SHADER_STORAGE_BLOCK_SIZE"]:
            raise ValueError(f"Scene with {count:,} Gaussians exceeds this OpenGL device's buffer limits")
        pixels = np.zeros((height * width, 4), dtype=np.float32)
        pixels[:count * stride] = scene.records.reshape(-1, 4)
        texture = self.context.texture((width, height), 4, pixels, dtype="f4")
        try:
            order = self.context.buffer(reserve=capacity * 8)
        except Exception:
            texture.release()
            raise
        if self.texture is not None:
            self.texture.release()
            self.order.release()
        self.texture, self.order = texture, order
        self.count, self.capacity, self.stride = count, capacity, stride
        self.degree = scene.sh_degree

    def draw(
        self,
        robot_camera: FloatArray,
        head: FloatArray,
        projection: FloatArray,
        target: moderngl.Framebuffer | None = None,
        *,
        clear: tuple[float, float, float, float] | None = (0.025, 0.035, 0.055, 1.0),
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
        self.context.viewport = (0, 0, *target.size)
        self.context.scissor = None
        self.context.enable_only(moderngl.BLEND)
        self.context.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.context.blend_equation = moderngl.FUNC_ADD
        if clear is not None:
            target.clear(*clear)
        if not self.count:
            return
        rotation = world_from_eye[:3, :3]
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation.T
        view[:3, 3] = -rotation.T @ world_from_eye[:3, 3]
        self.texture.use(0)
        self.order.bind_to_storage_buffer(0)
        self.keys["view"].write(view.T.copy())
        self.keys["count"] = self.count
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
        self.program["viewport"] = target.size
        self.program["stride"] = self.stride
        self.program["sh_degree"] = self.degree
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4, instances=self.count)

    def close(self) -> None:
        for resource in (self.vao, self.program, self.sort, self.keys, self.texture, self.order):
            if resource is not None:
                resource.release()
