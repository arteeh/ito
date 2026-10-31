"""The ToF mesh pipeline, without processes or windows (the worker and the tape replay both
drive it):

  driver Motion samples  -> PoseTrack (the robot's own camera pose: contact odometry for
                            position, trunk IMU + head joint encoders for orientation)
  head IMU samples       -> the camera's tilt (its own gravity fusion); heading stays the pose
                            stream's, since the head IMU's yaw is arbitrary
  ToF RangeGrid          -> pose at the grid's own time -> Volume.integrate (TSDF)
  camera frame           -> pose at its exposure       -> Volume.paint (voxel colours)
                                                        -> projector frames for the renderer
  dirty blocks           -> surface.extract            -> one mesh, published a few times a second

Nothing here sees the simulator's ground truth: every pose comes from the driver's stream.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import surface
from .geometry import PoseTrack, angle_between, rotation
from .volume import Volume
from .zones import zones_from_grid

# A sensor reading waits this long for the pose stream to reach its time, then is dropped.
POSE_WAIT = 0.4
PAINT_INTERVAL = 0.15
MESH_INTERVAL = 0.2
# Colour-only changes (repainted voxels) reach the mesh this often; the live projector already
# shows the current view at camera resolution, so the coarse vertex colours can lag.
RECOLOUR_INTERVAL = 1.5
LIVE_INTERVAL = 1 / 20
# A frame becomes a kept projector when it looks this differently from every kept one.
KEY_ANGLE = np.radians(14)
KEY_DISTANCE = 0.25


@dataclass
class Projector:
    slot: int
    image: np.ndarray | None  # (h, w, 3) uint8; None: unchanged image, new pose
    world_from_camera: np.ndarray
    intrinsics: tuple  # fx, fy, cx, cy, width, height of `image`
    time: float


@dataclass
class MeshFrame:
    """One publication: the whole mesh when it changed (else None) and changed projectors."""

    kind = "mesh"
    revision: int
    captured_at: float | None
    positions: np.ndarray | None = None
    normals: np.ndarray | None = None
    colours: np.ndarray | None = None
    triangles: np.ndarray | None = None
    projectors: list = field(default_factory=list)
    camera: np.ndarray | None = None  # latest posed camera, world_from_camera
    epoch: float = 0.0

    @property
    def observations(self):
        """For the windows' performance HUD: a new mesh made visible the ToF frame of
        `captured_at` (pilot clock), so "capture to visible" is ToF range to screen."""
        if self.positions is None or self.captured_at is None:
            return ()
        return ((self.revision, self.captured_at),)


# A head IMU reading tilts a pose only within this many seconds of it.
TILT_MATCH = 0.03
UP = np.array([0.0, 1.0, 0.0])


def tilted(camera, up):
    """camera (4x4 world_from_camera) turned the least so that world up, seen from the camera,
    is `up` (a unit vector in camera axes)."""
    a = camera[:3, :3].T @ UP  # world up in camera axes, as the pose has it
    axis = np.cross(up, a)
    s, c = np.linalg.norm(axis), float(np.clip(up @ a, -1, 1))
    if s < 1e-9:
        return camera
    k = axis / s
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    turn = np.eye(3) + s * skew + (1 - c) * skew @ skew  # rotates `up` onto `a`
    out = camera.copy()
    out[:3, :3] = camera[:3, :3] @ turn
    return out


class Fusion:
    def __init__(self, voxel=0.04, capacity=6000, keyframes=12, max_range=4.0, head_imu=True):
        self.volume = Volume(voxel=voxel, capacity=capacity, max_range=max_range)
        self.poses = PoseTrack()
        self.head_imu = head_imu
        self.tilts = deque(maxlen=500)  # (time, world up in camera axes) from the head IMU
        self.grids = []
        self.frames = []
        self.meshes = {}
        self.keyframes = keyframes  # projector slots: 0 live, 1.. kept views
        self.kept = {}  # slot -> (world_from_camera, time)
        self.next_kept = 1
        self.pending_projectors = []
        self.last_paint = -1e9
        self.last_live = -1e9
        self.last_mesh = -1e9
        self.last_recolour = -1e9
        self.revision = 0
        self.epoch = time.time()
        self.captured_at = None
        self.camera = None
        self.mesh_changed = False
        self.stats = dict(
            tof_frames=0, tof_dropped=0, tilted=0, tof_voxels=0, painted=0, frames=0,
            frames_dropped=0,
            vertices=0, triangles=0, blocks=0, evictions=0, fuse_ms=0.0, paint_ms=0.0,
            mesh_ms=0.0,
        )  # fmt: skip

    def motion(self, samples):
        for when, world_from_camera in samples:
            self.poses.add(when, world_from_camera)

    def imu(self, samples):
        """Head IMU samples with the chip's own orientation: [(pilot-clock time, sensor-to-world
        xyzw, sensor-to-camera 3x3), ...]. Only gravity (tilt) is taken from them."""
        for when, orientation, mount in samples:
            if self.tilts and when <= self.tilts[-1][0]:
                continue
            up = np.asarray(mount) @ rotation(orientation).T @ UP
            self.tilts.append((when, up / np.linalg.norm(up)))

    def pose(self, when, slack=0.1):
        """The camera at `when`: the pose stream, its tilt from the head IMU where one is near."""
        camera = self.poses.at(when, slack=slack)
        if camera is None or not self.head_imu or not self.tilts:
            return camera
        times = [t for t, _ in self.tilts]
        i = int(np.searchsorted(times, when))
        near = [self.tilts[j] for j in (i - 1, i) if 0 <= j < len(self.tilts)]
        near = [(t, u) for t, u in near if abs(t - when) <= TILT_MATCH]
        if not near:
            return camera
        if len(near) == 2 and near[1][0] > near[0][0]:
            f = (when - near[0][0]) / (near[1][0] - near[0][0])
            up = near[0][1] * (1 - f) + near[1][1] * f
        else:
            up = min(near, key=lambda n: abs(n[0] - when))[1]
        self.stats["tilted"] += 1
        return tilted(camera, up / np.linalg.norm(up))

    def ranges(self, grids):
        self.grids.extend(grids)
        del self.grids[:-32]

    def frame(self, image, exposed, intrinsics, fallback=None):
        """A camera frame for colour: image (h, w, 3) uint8 at the rectified pinhole
        `intrinsics` (fx, fy, cx, cy), exposed at pilot-clock `exposed`. `fallback` is the
        driver's own pose for this frame, used when the pose stream does not cover it."""
        self.frames.append((image, exposed, intrinsics, fallback))
        del self.frames[:-4]

    def _position(self):
        latest = self.poses.latest
        pose = self.poses.at(latest) if latest is not None else None
        return pose[:3, 3] if pose is not None else np.zeros(3)

    def step(self, now):
        """Fuse whatever the pose stream now covers; returns a MeshFrame when there is news."""
        latest = self.poses.latest
        waiting = []
        for grid in self.grids:
            if latest is None or grid["time"] > latest:
                if now - grid["time"] > POSE_WAIT:
                    self.stats["tof_dropped"] += 1
                else:
                    waiting.append(grid)
                continue
            camera = self.pose(grid["time"])
            if camera is None:
                self.stats["tof_dropped"] += 1
                continue
            started = time.perf_counter()
            zones = zones_from_grid(grid)
            world_from_sensor = camera @ zones.mount
            fused = self.volume.integrate(zones, world_from_sensor, grid["time"], camera[:3, 3])
            self.stats["tof_frames"] += 1
            self.stats["tof_voxels"] = fused
            self.stats["fuse_ms"] = (time.perf_counter() - started) * 1000
            self.captured_at = grid["time"]
        self.grids = waiting

        frames, self.frames = self.frames, []
        for image, exposed, intrinsics, fallback in frames:
            camera = self.pose(exposed, slack=0.05)
            if camera is None and latest is not None and exposed > latest:
                if now - exposed <= POSE_WAIT:
                    self.frames.append((image, exposed, intrinsics, fallback))
                    continue
            if camera is None:
                camera = fallback
            if camera is None:
                self.stats["frames_dropped"] += 1
                continue
            self.stats["frames"] += 1
            self.camera = camera
            self._project(image, camera, intrinsics, exposed)
            if exposed - self.last_paint >= PAINT_INTERVAL:
                self.last_paint = exposed
                started = time.perf_counter()
                self.stats["painted"] = self.volume.paint(image, camera, intrinsics, exposed)
                self.stats["paint_ms"] = (time.perf_counter() - started) * 1000

        news = None
        if now - self.last_mesh >= MESH_INTERVAL and (self.volume.dirty or self.volume.evicted):
            self.last_mesh = now
            colours = now - self.last_recolour >= RECOLOUR_INTERVAL
            if colours:
                self.last_recolour = now
            self._remesh(colours)
        if self.mesh_changed or self.pending_projectors:
            self.revision += 1
            news = MeshFrame(self.revision, self.captured_at, camera=self.camera, epoch=self.epoch)
            if self.mesh_changed:
                news.positions, news.normals, news.colours, news.triangles = self.combined()
                self.mesh_changed = False
            news.projectors, self.pending_projectors = self.pending_projectors, []
        return news

    def _project(self, image, camera, intrinsics, exposed):
        fx, fy, cx, cy = intrinsics
        h, w = image.shape[:2]
        k = (fx, fy, cx, cy, w, h)
        if exposed - self.last_live >= LIVE_INTERVAL:
            self.last_live = exposed
            self.pending_projectors = [p for p in self.pending_projectors if p.slot != 0]
            self.pending_projectors.append(Projector(0, image, camera, k, exposed))
        if self.keyframes <= 1:
            return
        for pose, _ in self.kept.values():
            if (
                angle_between(pose[:3, :3], camera[:3, :3]) < KEY_ANGLE
                and np.linalg.norm(pose[:3, 3] - camera[:3, 3]) < KEY_DISTANCE
            ):
                return
        if len(self.kept) < self.keyframes - 1:
            slot = self.next_kept
            self.next_kept += 1
        else:
            slot = min(self.kept, key=lambda s: self.kept[s][1])  # the oldest view
        self.kept[slot] = (camera, exposed)
        self.pending_projectors.append(Projector(slot, image, camera, k, exposed))

    def _remesh(self, colours):
        started = time.perf_counter()
        dirty, painted, evicted = self.volume.take_dirty(colours)
        for key in evicted:
            self.meshes.pop(key, None)
        # A block's mesh reaches one voxel into its neighbours: new geometry re-meshes them too.
        todo = set(dirty.tolist()) | set(painted.tolist())
        for slot in dirty:
            kx, ky, kz = self.volume.keys[slot]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        s = self.volume.slots.get((kx + dx, ky + dy, kz + dz))
                        if s is not None:
                            todo.add(s)
        slots = np.array(sorted(todo), np.int64)
        for start in range(0, len(slots), 256):
            for key, pos, nrm, col, tri in surface.extract(self.volume, slots[start : start + 256]):
                if pos is None:
                    self.meshes.pop(key, None)
                else:
                    self.meshes[key] = (pos, nrm, col, tri)
        self.mesh_changed = True
        self.stats["mesh_ms"] = (time.perf_counter() - started) * 1000
        self.stats["blocks"] = self.volume.used
        self.stats["evictions"] = self.volume.evictions

    def combined(self):
        parts = list(self.meshes.values())
        if not parts:
            empty = np.zeros((0, 3), np.float32)
            return empty, empty, np.zeros((0, 4), np.uint8), np.zeros((0, 3), np.uint32)
        counts = np.array([len(p[0]) for p in parts])
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint32)
        positions = np.concatenate([p[0] for p in parts])
        normals = np.concatenate([p[1] for p in parts])
        colours = np.concatenate([p[2] for p in parts])
        triangles = np.concatenate([p[3] + o for p, o in zip(parts, offsets, strict=True)])
        self.stats["vertices"] = len(positions)
        self.stats["triangles"] = len(triangles)
        return positions, normals, colours, triangles
