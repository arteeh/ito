"""Draws MASt3R-SLAM's Gaussian splats: main's OpenGL GaussianRenderer (ito/render/renderer.py at
6426a50) on Ito's Vulkan (ito.render.vk), with the same shaders (ito/shaders/splat*).

Splats live in fixed slots of one storage buffer, four vec4 records each ([xyz, opacity],
[scale, fade end], [wxyz rotation], [SH colour, retiring]); a ring.SplatUpdate replaces the
listed slots (zero opacity frees one) and its retiring splats fade out over the fade time. The
slots are sorted back to front on the GPU once per displayed frame, from the viewer between
the pilot's eyes (a bitonic sort: 1024-key tiles in shared memory, then global merges), and
every view draws them over each other in that order, premultiplied.
"""

import numpy as np
import vulkan as vk

from ito import clock
from ito.render.pose import validate_pose
from ito.render.vk import Buffer, Pipeline, mat, pack
from ito.tofmesh.render import CLEAR, COLOUR

STORAGE = (
    vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
)
RECORD = 64  # bytes per splat: four vec4
TILE = 1024  # the sort's tile: its shaders' workgroup


def _view(world_from_eye):
    rotation = world_from_eye[:3, :3]
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ world_from_eye[:3, 3]
    return view


def _sort_size(count):
    return max(TILE, 1 << max(0, count - 1).bit_length())


class SplatRenderer:
    """The splats on the GPU. Its work goes into the device's frame being recorded
    (vk.Device.frame): upload() and prepare() once per displayed frame, draw() once per view."""

    def __init__(self, device):
        self.device = device
        self.presort = Pipeline(
            device, ("splat_presort.comp",), {0: "storage", 1: "storage"}, push=72
        )
        self.merge = Pipeline(device, ("splat_sort.comp",), {0: "storage"}, push=16)
        self.program = Pipeline(
            device, ("splat.vert", "splat.frag"), {0: "storage", 1: "storage", 2: "uniform"},
            colours=(COLOUR,), blend="premultiplied",
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        )  # fmt: skip
        self.scene = self.order = None
        self.live = None  # which slots hold a splat
        self.slot_capacity = 0
        self.draw_count = 0  # slots drawn: up to the last live one
        self.capacity = TILE  # keys sorted: draw_count up to a power of two
        self.count = 0
        self.epoch = None
        self.fade_seconds = 0.0
        self.uploaded_bytes = 0
        self.sorts = 0

    # -- updates -------------------------------------------------------------------------
    def _allocate(self, capacity, *, keep):
        frame = self.device.frame
        scene = Buffer(self.device, capacity * RECORD, STORAGE)
        order = Buffer(self.device, _sort_size(capacity) * 8, STORAGE)
        frame.fill(scene)
        live = np.zeros(capacity, dtype=bool)
        if self.scene is not None:
            if keep:
                kept = min(capacity, self.slot_capacity)
                frame.barrier()
                frame.copy(self.scene, scene, kept * RECORD)
                live[:kept] = self.live[:kept]
            self.device.defer(self.scene, self.order)
        self.scene, self.order, self.live = scene, order, live
        self.slot_capacity = capacity
        if not keep:
            self.draw_count = 0
        elif self.draw_count > capacity:
            occupied = np.flatnonzero(live)
            self.draw_count = int(occupied[-1]) + 1 if len(occupied) else 0
        self.capacity = _sort_size(self.draw_count)

    def upload(self, update):
        """Apply one ring.SplatUpdate."""
        n = len(update.indices)
        if n > 4096:
            raise ValueError("Splat updates are limited to 4096 records per packet")
        frame = self.device.frame
        new_scene = self.epoch != update.epoch
        if self.live is None or update.capacity != self.slot_capacity:
            self._allocate(update.capacity, keep=bool(self.fade_seconds and not new_scene))
        elif not self.fade_seconds or new_scene:
            frame.fill(self.scene)
            self.live.fill(False)
            self.draw_count = 0
            self.capacity = TILE
        self.count = update.count
        self.epoch, self.fade_seconds = update.epoch, update.fade_seconds
        if n:
            self.live[update.indices] = update.records[:, 0, 3] > 0
            self.draw_count = max(self.draw_count, int(update.indices.max()) + 1)
            if not self.live[self.draw_count - 1]:
                occupied = np.flatnonzero(self.live[: self.draw_count])
                self.draw_count = int(occupied[-1]) + 1 if len(occupied) else 0
            self.capacity = _sort_size(self.draw_count)
            records = update.records.copy()
            retiring = records[:, 3, 3] == 1
            records[retiring, 1, 3] = np.minimum(
                records[retiring, 1, 3], clock.now() - self.epoch + self.fade_seconds
            )
            frame.barrier()
            frame.scatter(
                self.scene,
                records.reshape(n, RECORD // 4),
                update.indices.astype(np.int64) * RECORD,
            )
            self.uploaded_bytes += update.records.nbytes + update.indices.nbytes
        if update.acknowledge is not None:
            update.acknowledge()

    def clear(self):
        """Nothing shown (the next session's scene starts empty)."""
        if self.scene is not None:
            self.device.defer(self.scene, self.order)
        self.scene = self.order = self.live = None
        self.slot_capacity = self.draw_count = self.count = 0
        self.capacity = TILE
        self.epoch = None
        self.fade_seconds = 0.0

    # -- drawing -------------------------------------------------------------------------
    def prepare(self, robot_camera, head, projection):
        """Once per displayed frame: the splats sorted back to front from the viewer, the point
        between the pilot's eyes (a desktop's one view)."""
        if not self.draw_count:
            return
        frame = self.device.frame
        view = _view(validate_pose(robot_camera) @ validate_pose(head))
        whole = (0, self.scene.size)
        frame.barrier()
        frame.use(
            self.presort,
            {0: (self.order, 0, self.order.size), 1: (self.scene, *whole)},
            pack("16f2I", mat(view), self.draw_count, RECORD // 16),
        )
        frame.dispatch((self.capacity, 1), group=TILE)
        frame.barrier()
        stage = 2 * TILE
        while stage <= self.capacity:
            distance = stage // 2
            while distance:
                local = distance < TILE
                frame.use(
                    self.merge,
                    {0: (self.order, 0, self.order.size)},
                    pack("4I", self.capacity, stage, distance, int(local)),
                )
                frame.dispatch((self.capacity, 1), group=TILE)
                frame.barrier()
                distance = 0 if local else distance // 2
            stage *= 2
        self.sorts += 1

    def draw(self, robot_camera, head, projection, target, *, clear=CLEAR):
        """One view of the frame prepare() sorted: the eye at head, with this projection, into
        target (an image of format COLOUR, OpenGL's row order: vk.Presenter)."""
        world_from_eye = validate_pose(robot_camera) @ validate_pose(head)
        projection = np.asarray(projection, dtype=np.float64)
        if projection.shape != (4, 4) or not np.isfinite(projection).all():
            raise ValueError("Projection must be a finite 4x4 matrix")
        frame = self.device.frame
        with frame.render([target], clear=clear or (0, 0, 0, 1)):
            if not self.draw_count:
                return
            scene_time = clock.now() - self.epoch if self.fade_seconds else 0.0
            block = pack(
                "16f16f4f4f4i", mat(_view(world_from_eye)), mat(projection),
                (*world_from_eye[:3, 3], scene_time), (*target.size, self.fade_seconds, 0.0),
                (RECORD // 16, 0, 0, 0),
            )  # fmt: skip
            frame.use(
                self.program,
                {
                    0: (self.order, 0, self.order.size),
                    1: (self.scene, 0, self.scene.size),
                    2: frame.uniform(block),
                },
            )
            frame.draw(4, instances=self.draw_count)

    def close(self):
        for resource in (self.presort, self.merge, self.program, self.scene, self.order):
            if resource is not None:
                resource.close()
