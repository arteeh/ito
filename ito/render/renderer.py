import logging
import os
from importlib.resources import files

import moderngl
import numpy as np
from OpenGL import GL

from ito import clock

from .pose import validate_pose
from .scene import FloatArray, GaussianBuffer

# ITO_EASE=<seconds> (an experiment, off by default): every change to a live splat glides
# from what is drawn to the new record over this time (new splats fade in), and a retired
# splat stays whole that long before its fade, so a replacement is in before it goes.
EASE = max(0.0, float(os.environ.get("ITO_EASE", "0") or 0))
# ITO_UPDATE_STATS=1 (a diagnostic): log every 5 s how the applied packets changed the live
# splats: born, retired, emptied, and how far the changed ones moved.
UPDATE_STATS = os.environ.get("ITO_UPDATE_STATS") == "1"
# ITO_AA=1 (training.py): draw as gsplat's "antialiased" mode trained the splats.
AA = os.environ.get("ITO_AA", "0") != "0"
log = logging.getLogger(__name__)


class UpdateStats:
    def __init__(self):
        self.mirror = np.zeros((0, 2, 4), np.float32)
        self.since = clock.now()
        self.reset()

    def reset(self):
        self.packets = self.born = self.retired = self.emptied = self.kept = 0
        self.moves = []

    def __call__(self, indices, records):
        if len(self.mirror) <= int(indices.max()):
            grown = np.zeros((max(1024, 2 * int(indices.max()) + 2), 2, 4), np.float32)
            grown[: len(self.mirror)] = self.mirror
            self.mirror = grown
        old = self.mirror[indices]
        live_before = (old[:, 0, 3] > 0) & (old[:, 1, 3] != 1)
        live_after = records[:, 0, 3] > 0
        retiring = records[:, 3, 3] == 1
        self.packets += 1
        self.born += int((~live_before & live_after & ~retiring).sum())
        self.retired += int((live_before & retiring).sum())
        self.emptied += int((~live_after).sum())
        changed = live_before & live_after & ~retiring
        self.kept += int(changed.sum())
        if changed.any():
            self.moves.append(np.linalg.norm(records[changed, 0, :3] - old[changed, 0, :3], axis=1))
        self.mirror[indices, 0] = records[:, 0]
        self.mirror[indices, 1, 3] = retiring
        now = clock.now()
        if now - self.since >= 5:
            moves = np.concatenate(self.moves) * 1000 if self.moves else np.zeros(1)
            log.info(
                "Splat updates in %.1f s: %d packets, %d born, %d retired, %d emptied, %d changed"
                " (moved p50 %.1f p95 %.1f max %.0f mm, %d over 1 cm)",
                now - self.since,
                self.packets,
                self.born,
                self.retired,
                self.emptied,
                self.kept,
                *np.percentile(moves, [50, 95, 100]),
                int((moves > 10).sum()),
            )
            self.since = now
            self.reset()


class GaussianRenderer:
    def __init__(self, context: moderngl.Context):
        if context.version_code < 430:
            raise RuntimeError("Ito needs OpenGL 4.3 compute shaders (Mesa llvmpipe is supported)")
        self.context = context
        shaders = files("ito.render").joinpath("shaders")
        self.keys = context.compute_shader(shaders.joinpath("presort.glsl").read_text())
        self.sort_program = context.compute_shader(shaders.joinpath("sort.glsl").read_text())
        self.program = context.program(
            vertex_shader=shaders.joinpath("splat.vert").read_text(),
            fragment_shader=shaders.joinpath("splat.frag").read_text(),
        )
        self.vao = context.vertex_array(self.program, [])
        self.update = context.compute_shader(shaders.joinpath("update.glsl").read_text())
        self.staging = [
            (context.buffer(reserve=4096 * 64), context.buffer(reserve=4096 * 4)) for _ in range(4)
        ]
        self.staging_fences = [None] * len(self.staging)
        self.staging_cursor = 0
        self.epoch = 0.0
        self.fade_seconds = 0.0
        self.uploaded_bytes = 0
        self.draw_count = 0
        self.sorts = 0
        self.scene_buffer = None
        self.previous = None
        self.ease = 0.0
        self.order = None
        self.count = 0
        self.capacity = 0
        self.slot_capacity = 0
        self.live = None
        self.observed_at = None
        self.stride = 4
        self.degree = 0
        self.revision = -1
        self.coordinate_frame = None
        self.stats = UpdateStats() if UPDATE_STATS else None
        self.color_gain = (0.0, 0.0, 0.0)

    def upload(self, scene: GaussianBuffer) -> None:
        count, stride, _ = scene.records.shape
        capacity = max(1024, 1 << max(0, count - 1).bit_length())
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
        self.draw_count = self.slot_capacity = count
        self.live = None
        self.fade_seconds = 0.0
        self.ease = 0.0
        if self.previous is not None:
            self.previous.release()
            self.previous = None
        self.uploaded_bytes += scene.records.nbytes

    def _allocate_live(self, capacity, *, keep):
        sort_capacity = max(1024, 1 << max(0, capacity - 1).bit_length())
        limit = self.context.info["GL_MAX_SHADER_STORAGE_BLOCK_SIZE"]
        if max(capacity * 64, sort_capacity * 8) > limit:
            raise ValueError("Splat budget exceeds OpenGL buffer limits")
        scene = self.context.buffer(reserve=capacity * 64)
        try:
            scene.clear()
            order = self.context.buffer(reserve=sort_capacity * 8)
        except Exception:
            scene.release()
            raise
        previous = None
        if EASE:
            previous = self.context.buffer(reserve=capacity * 64)
            previous.clear()
        live = np.zeros(capacity, dtype=bool)
        observed_at = np.zeros(capacity, dtype=np.float64)
        if self.scene_buffer is not None:
            if keep:
                kept = min(capacity, self.slot_capacity)
                self.context.copy_buffer(scene, self.scene_buffer, size=kept * 64)
                if previous is not None and self.previous is not None:
                    self.context.copy_buffer(previous, self.previous, size=kept * 64)
                live[:kept] = self.live[:kept]
                observed_at[:kept] = self.observed_at[:kept]
            self.scene_buffer.release()
            self.order.release()
        if self.previous is not None:
            self.previous.release()
        self.previous, self.ease = previous, EASE
        self.scene_buffer, self.order, self.live = scene, order, live
        self.observed_at = observed_at
        self.slot_capacity = capacity
        if not keep:
            self.draw_count = 0
        elif self.draw_count > capacity:
            occupied = np.flatnonzero(live)
            self.draw_count = int(occupied[-1]) + 1 if len(occupied) else 0
        self.capacity = max(1024, 1 << max(0, self.draw_count - 1).bit_length())

    def apply(self, update) -> bool:
        if update.epoch < self.epoch or (
            update.epoch == self.epoch and update.revision <= self.revision
        ):
            if update.acknowledge is not None:
                update.acknowledge()
            return True
        if update.epoch == self.epoch and self.coordinate_frame not in (
            None,
            update.coordinate_frame,
        ):
            raise ValueError("A coordinate frame change requires a new complete scene")
        n = len(update.indices)
        if n > 4096:
            raise ValueError("Splat updates are limited to 4096 records per packet")
        slot = self.staging_cursor
        fence = self.staging_fences[slot]
        if n and fence is not None:
            status = GL.glClientWaitSync(fence, GL.GL_SYNC_FLUSH_COMMANDS_BIT, 0)
            if status == GL.GL_TIMEOUT_EXPIRED:
                return False
            if status == GL.GL_WAIT_FAILED:
                raise RuntimeError("Could not check the Gaussian upload fence")
            GL.glDeleteSync(fence)
            self.staging_fences[slot] = None
        required = update.capacity
        new_scene = self.epoch != update.epoch
        if self.live is None or required != self.slot_capacity:
            self._allocate_live(required, keep=bool(self.fade_seconds and not new_scene))
        elif not self.fade_seconds or new_scene:
            self.scene_buffer.clear()
            if self.previous is not None:
                self.previous.clear()
            self.live.fill(False)
            self.observed_at.fill(0)
            self.draw_count = 0
            self.capacity = 1024
        self.count = update.count
        self.stride, self.degree = 4, 0
        self.epoch, self.fade_seconds = update.epoch, update.fade_seconds
        self.color_gain = getattr(update, "color_gain", (0.0, 0.0, 0.0))
        self.revision = update.revision
        self.coordinate_frame = update.coordinate_frame
        if not n:
            if update.acknowledge is not None:
                update.acknowledge()
            return True
        self.live[update.indices] = update.records[:, 0, 3] > 0
        self.observed_at[update.indices] = (
            update.observed_at if update.observed_at is not None else update.captured_at
        )
        self.draw_count = max(self.draw_count, int(update.indices.max()) + 1)
        if not self.live[self.draw_count - 1]:
            occupied = np.flatnonzero(self.live[: self.draw_count])
            self.draw_count = int(occupied[-1]) + 1 if len(occupied) else 0
        self.capacity = max(1024, 1 << max(0, self.draw_count - 1).bit_length())
        if self.stats is not None:
            self.stats(update.indices, update.records)
        changes, slots = self.staging[slot]
        records = update.records.copy()
        retiring = records[:, 3, 3] == 1
        records[retiring, 1, 3] = np.minimum(
            records[retiring, 1, 3], clock.now() - self.epoch + self.fade_seconds + self.ease
        )
        changes.write(records)
        slots.write(update.indices)
        self.uploaded_bytes += update.records.nbytes + update.indices.nbytes
        self.scene_buffer.bind_to_storage_buffer(1)
        changes.bind_to_storage_buffer(2)
        slots.bind_to_storage_buffer(3)
        (self.scene_buffer if self.previous is None else self.previous).bind_to_storage_buffer(4)
        self.update["ease"] = self.ease
        self.update["scene_time"] = clock.now() - self.epoch
        self.update["count"] = n
        self.update.run(group_x=(n + 255) // 256)
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        # PyOpenGL casts opaque pointer arguments through ctypes, creating a
        # self-referencing ownership dictionary per fence. Keep its address:
        # the driver still owns the sync, without triggering full Python GC
        # scans in display calls after thousands of packets.
        self.staging_fences[slot] = GL.glFenceSync(GL.GL_SYNC_GPU_COMMANDS_COMPLETE, 0).address
        self.staging_cursor = (slot + 1) % len(self.staging)
        if update.acknowledge is not None:
            update.acknowledge()
        return True

    def sort(self, robot_camera: FloatArray, head: FloatArray) -> None:
        if not self.draw_count:
            return
        view = self._view(validate_pose(robot_camera) @ validate_pose(head))
        self.scene_buffer.bind_to_storage_buffer(1)
        self.order.bind_to_storage_buffer(0)
        self.keys["view"].write(view.T.copy())
        self.keys["count"] = self.draw_count
        self.keys["stride"] = self.stride
        self.keys.run(group_x=self.capacity // 1024)
        # A software GPU can block inside a barrier. ctypes releases the GIL so
        # tracking and the deadman sender keep running while the GPU catches up.
        GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        self.sort_program["capacity"] = self.capacity
        stage = 2048
        while stage <= self.capacity:
            self.sort_program["stage"] = stage
            distance = stage // 2
            while distance:
                local = distance < 1024
                self.sort_program["distance"] = distance
                self.sort_program["local_merge"] = local
                self.sort_program.run(group_x=self.capacity // 1024)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                distance = 0 if local else distance // 2
            stage *= 2
        self.sorts += 1

    @staticmethod
    def _view(world_from_eye: FloatArray) -> FloatArray:
        rotation = world_from_eye[:3, :3]
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation.T
        view[:3, 3] = -rotation.T @ world_from_eye[:3, 3]
        return view

    def draw(
        self,
        robot_camera: FloatArray,
        head: FloatArray,
        projection: FloatArray,
        target: moderngl.Framebuffer | None = None,
        *,
        clear: tuple[float, float, float, float] | None = (0.025, 0.035, 0.055, 1.0),
        viewport: tuple[int, int, int, int] | None = None,
        sort: bool = True,
    ) -> None:
        """Draw world-space splats from world_from_camera @ camera_from_head.

        Call once per eye with its current pose, projection and framebuffer. Scene
        updates and anchor updates are independent; a stalled producer never stalls
        tracking. Projection uses OpenGL's [-1, 1] clip depth. Owns GL draw state.
        With sort=False the order from the last sort() is reused.
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
        if sort:
            self.sort(robot_camera, head)
        view = self._view(world_from_eye)
        self.scene_buffer.bind_to_storage_buffer(1)
        (self.scene_buffer if self.previous is None else self.previous).bind_to_storage_buffer(4)
        self.order.bind_to_storage_buffer(0)
        self.program["view"].write(view.T.copy())
        self.program["projection"].write(projection.T.copy())
        self.program["eye"] = tuple(world_from_eye[:3, 3])
        self.program["viewport"] = viewport[2:]
        self.program["scene_time"] = (
            clock.now() - self.epoch if self.fade_seconds or self.ease else 0
        )
        self.program["fade_seconds"] = self.fade_seconds
        self.program["ease"] = self.ease
        self.program["antialiased"] = int(AA and self.live is not None)
        self.program["color_gain"] = (
            tuple(np.exp(self.color_gain)) if self.live is not None else (1, 1, 1)
        )
        self.program["stride"] = self.stride
        self.program["sh_degree"] = self.degree
        self.vao.render(mode=moderngl.TRIANGLE_STRIP, vertices=4, instances=self.draw_count)

    def close(self) -> None:
        for fence in self.staging_fences:
            if fence is not None:
                GL.glDeleteSync(fence)
        for changes, slots in self.staging:
            changes.release()
            slots.release()
        for resource in (
            self.vao,
            self.program,
            self.sort_program,
            self.keys,
            self.scene_buffer,
            self.previous,
            self.order,
            self.update,
        ):
            if resource is not None:
                resource.release()
