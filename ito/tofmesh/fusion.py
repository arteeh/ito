"""The ToF mesh pipeline, without processes or windows (the worker and the tape replay both
drive it):

  driver Motion samples  -> PoseTrack (the robot's own camera pose: contact odometry for
                            position, trunk IMU + head joint encoders for orientation)
  head IMU samples       -> the camera's orientation (its chip's own fusion): every quick turn,
                            sway and gear-play wobble of the head; its arbitrary heading is tied
                            to the pose stream's by a slow average of their difference
  ToF RangeGrid          -> pose at the grid's own time -> Volume.integrate (TSDF)
  camera frame           -> pose at its exposure (arrival minus the driver's capture_delay
                            and the delay measured from the frames, latency.FrameDelay)
                                                        -> Volume.paint (voxel colours)
                                                        -> projector frames for the renderer
  dirty blocks           -> surface.extract            -> one mesh, published a few times a second

Nothing here sees the simulator's ground truth: every pose comes from the driver's stream.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import surface
from . import zones as zones_module
from .geometry import PoseTrack, angle_between, quaternion, rotation, slerp
from .latency import FrameDelay
from .volume import Volume
from .zones import zones_from_grid

# A sensor reading waits this long for the pose stream to reach its time, then is dropped.
POSE_WAIT = 0.4
PAINT_INTERVAL = 0.15
# New geometry is re-meshed at once, after every fused ToF frame (~12.7 Hz), but not more often
# than this: waiting for a slower re-mesh tick was up to 0.2 s of the mesh's lag.
MESH_INTERVAL = 0.05
# Colour-only changes (repainted voxels) reach the mesh this often; the live projector already
# shows the current view at camera resolution, so the coarse vertex colours can lag.
RECOLOUR_INTERVAL = 1.5
# ... spread over the re-meshes that follow, this many blocks each, so a repaint of most of the
# view (the floor) does not stall the next ToF frame behind one long re-mesh.
RECOLOUR_BATCH = 48
LIVE_INTERVAL = 1 / 20
# Projector slots: 0 the live frame, the rest kept views (the renderer's texture array).
PROJECTORS = 12
# A frame becomes a kept projector when it looks this differently from every kept one.
KEY_ANGLE = np.radians(14)
KEY_DISTANCE = 0.25
# A kept view this old is replaced by a new frame looking about the same way, so a changed scene
# (an object gone) is not projected from an old picture of it.
KEY_REFRESH = 3.0
# A ToF frame's edges are placed along the camera frame nearest to it in time (zones.guide), if
# one is this near. Frames reach the worker ~0.1 s after their exposure, ToF frames sooner, and
# a ToF frame does not wait for its frame (that would add the video's delay to the mesh's): it
# takes the newest one, projected with the poses of both moments, so only things that moved in
# between land off their edge.
GUIDE_MATCH = 0.2


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
    # Where the time went for the ToF frame of `captured_at` (diagnostics): see Fusion.step.
    timing: dict | None = None

    @property
    def observations(self):
        """For the windows' performance HUD: a new mesh made visible the ToF frame of
        `captured_at` (pilot clock), so "capture to visible" is ToF range to screen."""
        if self.positions is None or self.captured_at is None:
            return ()
        return ((self.revision, self.captured_at),)


# A head IMU reading orients a pose only within this many seconds of it.
TILT_MATCH = 0.03
# Without one that near, the last correction holds for this long, so the map never mixes
# corrected and uncorrected poses.
TILT_HOLD = 0.5
# The head IMU's heading is arbitrary and drifts; the pose stream's (trunk IMU and head joint
# encoders) is anchored but carries the head's gear play, ~1 degree per servo that flips with
# the gait. The IMU's heading is tied to the pose stream's by their difference averaged over
# this many seconds (a first-order low-pass): the IMU keeps every quick movement, the pose
# stream keeps where the head points over seconds. Measured in the sim (BigKangaroo, 1 Nov):
# camera-pose error that changes within a second, walking, 1.13 -> 0.44 degrees median.
HEADING_SECONDS = 1.0
UP = np.array([0.0, 1.0, 0.0])


def heading_of(r):
    """Angle (radians) about world +Y of a 3x3 rotation's view direction (-Z)."""
    ahead = r @ np.array([0.0, 0.0, -1.0])
    return float(np.arctan2(-ahead[0], -ahead[2]))


def about_up(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tilt(camera, up):
    """The least rotation (3x3, in camera axes) that turns camera (4x4 world_from_camera) so
    that world up, seen from the camera, is `up` (a unit vector in camera axes)."""
    a = camera[:3, :3].T @ UP  # world up in camera axes, as the pose has it
    axis = np.cross(up, a)
    s, c = np.linalg.norm(axis), float(np.clip(up @ a, -1, 1))
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * skew + (1 - c) * skew @ skew  # rotates `up` onto `a`


def turned(camera, turn):
    out = camera.copy()
    out[:3, :3] = camera[:3, :3] @ turn
    return out


class Fusion:
    def __init__(
        self, voxel=0.04, capacity=6000, keyframes=PROJECTORS, max_range=4.0, head_imu=True
    ):
        self.volume = Volume(voxel=voxel, capacity=capacity, max_range=max_range)
        self.poses = PoseTrack()
        self.head_imu = head_imu
        # (time, xyzw): the camera's orientation in the head IMU's world (+Y up, any heading)
        self.orientations = deque(maxlen=500)
        self.unheaded = deque(maxlen=500)  # pose-stream samples (time, 3x3) not yet compared
        self.headings = deque(maxlen=500)  # (time, radians): pose-stream minus IMU heading
        self.last_turn = None  # (time, 3x3): the latest correction made
        self.grids = []
        self.frames = []
        self.frame_delay = FrameDelay()
        self.frame_intrinsics = None
        self.posed = deque(maxlen=6)  # recent posed frames: (exposed, image, camera, intrinsics)
        self.meshes = {}
        self.keyframes = keyframes  # projector slots: 0 live, 1.. kept views
        self.kept = {}  # slot -> (world_from_camera, time)
        self.next_kept = 1
        self.pending_projectors = []
        self.last_paint = -1e9
        self.last_live = -1e9
        self.last_mesh = -1e9
        self.last_recolour = -1e9
        self.recolour = set()  # repainted blocks still to re-mesh
        self.revision = 0
        self.epoch = time.time()
        self.captured_at = None
        self.timing = None  # the newest fused ToF frame's stage times, until it is meshed
        self.meshed_timing = None  # ... and once meshed, until published
        self.camera = None
        self.mesh_changed = False
        self.stats = dict(
            tof_frames=0, tof_dropped=0, tilted=0, tilt_held=0, untilted=0, tof_voxels=0,
            tilt_ahead=0, painted=0, frames=0, frames_dropped=0, frame_delay_extra_ms=0.0,
            guided=0, unguided=0, guide_ms=0.0,
            vertices=0, triangles=0, blocks=0, evictions=0, fuse_ms=0.0, paint_ms=0.0,
            mesh_ms=0.0,
        )  # fmt: skip

    def motion(self, samples):
        for when, world_from_camera in samples:
            self.poses.add(when, world_from_camera)
            if self.head_imu:
                self.unheaded.append((when, np.asarray(world_from_camera)[:3, :3]))
        self._compare_headings()

    def imu(self, samples):
        """Head IMU samples with the chip's own orientation: [(pilot-clock time, sensor-to-world
        xyzw, sensor-to-camera 3x3), ...], world +Y up, heading arbitrary."""
        for when, orientation, mount in samples:
            if self.orientations and when <= self.orientations[-1][0]:
                continue
            camera = rotation(orientation) @ np.asarray(mount, np.float64).T
            self.orientations.append((when, quaternion(camera)))
        self._compare_headings()

    def _orientation(self, when):
        """The camera's orientation in the IMU's world at `when` (3x3), or None past the newest
        reading or without readings within TILT_MATCH on both sides (one, if it is that near)."""
        o = self.orientations
        if not o or when > o[-1][0] or when < o[0][0] - TILT_MATCH:
            return None
        lo, hi = 0, len(o)
        while lo < hi:  # first reading at or after `when`
            mid = (lo + hi) // 2
            if o[mid][0] < when:
                lo = mid + 1
            else:
                hi = mid
        after = o[lo] if lo < len(o) else None
        before = o[lo - 1] if lo > 0 else None
        if before and after and after[0] - before[0] <= 2 * TILT_MATCH and after[0] > before[0]:
            f = (when - before[0]) / (after[0] - before[0])
            return rotation(slerp(before[1], after[1], f))
        near = min((r for r in (before, after) if r), key=lambda r: abs(r[0] - when))
        return rotation(near[1]) if abs(near[0] - when) <= TILT_MATCH else None

    def _compare_headings(self):
        """Fold pose-stream samples the IMU now covers into the heading difference (low-pass)."""
        if not self.orientations:
            return
        covered = self.orientations[-1][0]
        while self.unheaded and self.unheaded[0][0] <= covered:
            when, stream = self.unheaded.popleft()
            imu = self._orientation(when)
            if imu is None:
                continue
            difference = heading_of(stream @ imu.T)
            if not self.headings or when - self.headings[-1][0] > TILT_HOLD:
                value = difference  # first, or after a gap: start over
            else:
                last_time, last = self.headings[-1]
                gain = 1.0 - np.exp(-(when - last_time) / HEADING_SECONDS)
                change = (difference - last + np.pi) % (2 * np.pi) - np.pi  # shortest way round
                value = last + gain * change
            self.headings.append((when, float(value)))

    def _heading(self, when):
        """The heading difference as averaged up to `when` (None without one within TILT_HOLD)."""
        h = self.headings
        if not h or when < h[0][0] - TILT_HOLD:
            return None
        lo, hi = 0, len(h)
        while lo < hi:  # first entry after `when`
            mid = (lo + hi) // 2
            if h[mid][0] <= when:
                lo = mid + 1
            else:
                hi = mid
        entry = h[lo - 1] if lo > 0 else h[0]
        return entry[1] if abs(entry[0] - when) <= TILT_HOLD else None

    def covered(self, when):
        """Whether every source of `when`'s pose has passed it: the pose stream, and the head
        IMU once it has spoken (its batches can arrive after the frames and grids)."""
        latest = self.poses.latest
        if latest is None or when > latest:
            return False
        return not (self.head_imu and self.orientations and when > self.orientations[-1][0])

    def pose(self, when, slack=0.1):
        """The camera at `when`: position from the pose stream; orientation from the head IMU
        where it has a reading near, its heading tied to the pose stream's (HEADING_SECONDS)."""
        camera = self.poses.at(when, slack=slack)
        if camera is None or not self.head_imu or not self.orientations:
            return camera
        turn = self._turn(when, camera)
        if turn is not None:
            self.stats["tilted"] += 1
            if self.last_turn is None or when >= self.last_turn[0]:
                self.last_turn = (when, turn)
            return turned(camera, turn)
        newest = self.orientations[-1][0]
        if newest < when <= newest + TILT_HOLD:
            # The IMU has not reached `when` yet (its batches can trail the ToF's): the
            # correction it gives at its newest reading, on the pose stream's movement since.
            then = self.poses.at(newest, slack=slack)
            turn = None if then is None else self._turn(newest, then)
            if turn is not None:
                self.stats["tilt_ahead"] += 1
                return turned(camera, turn)
        if self.last_turn is not None and abs(when - self.last_turn[0]) <= TILT_HOLD:
            self.stats["tilt_held"] += 1
            return turned(camera, self.last_turn[1])
        self.stats["untilted"] += 1
        return camera

    def _turn(self, when, camera):
        """The correction (3x3, camera axes) that turns the pose stream's camera at `when` into
        the IMU's orientation there, or None without an IMU reading."""
        imu = self._orientation(when)
        if imu is None:
            return None
        heading = self._heading(when)
        if heading is None:
            # No heading comparison yet: the IMU's tilt only, the pose stream's heading.
            up = imu.T @ UP
            return tilt(camera, up / np.linalg.norm(up))
        return camera[:3, :3].T @ about_up(heading) @ imu

    def ranges(self, grids):
        self.grids.extend(grids)
        del self.grids[:-32]

    def frame(self, image, exposed, intrinsics, fallback=None):
        """A camera frame for colour: image (h, w, 3) uint8 at the rectified pinhole
        `intrinsics` (fx, fy, cx, cy), exposed at pilot-clock `exposed`. `fallback` is the
        driver's own pose for this frame, used when the pose stream does not cover it."""
        self.frames.append((image, exposed, intrinsics, fallback))
        del self.frames[:-4]
        self.frame_delay.frame(image, exposed)
        self.frame_intrinsics = intrinsics

    def step(self, now):
        """Fuse whatever the pose stream now covers; returns a MeshFrame when there is news."""
        latest = self.poses.latest
        # Frames first, so a ToF frame below finds the camera frames posed up to its time.
        self._measure_frame_delay(now)
        extra = self.frame_delay.extra
        frames, self.frames = self.frames, []
        for image, stamped, intrinsics, fallback in frames:
            exposed = stamped - extra
            # Wait for the pose stream to pass the exposure rather than hold its end pose.
            if not self.covered(exposed) and now - exposed <= POSE_WAIT:
                self.frames.append((image, stamped, intrinsics, fallback))
                continue
            camera = self.pose(exposed, slack=0.05)
            if camera is None:
                camera = fallback
            if camera is None:
                self.stats["frames_dropped"] += 1
                continue
            self.stats["frames"] += 1
            self.camera = camera
            self.posed.append((exposed, image, camera, intrinsics))
            self._project(image, camera, intrinsics, exposed)
            if exposed - self.last_paint >= PAINT_INTERVAL:
                self.last_paint = exposed
                started = time.perf_counter()
                self.stats["painted"] = self.volume.paint(image, camera, intrinsics, exposed)
                self.stats["paint_ms"] = (time.perf_counter() - started) * 1000

        waiting = []
        for grid in self.grids:
            # Grids wait for the pose stream only: the head IMU's batches can trail them by tens
            # of milliseconds, which would add to the mesh's lag; pose() bridges that gap.
            if (latest is None or grid["time"] > latest) and now - grid["time"] <= POSE_WAIT:
                waiting.append(grid)
                continue
            camera = None if latest is None or grid["time"] > latest else self.pose(grid["time"])
            if camera is None:
                self.stats["tof_dropped"] += 1
                continue
            started = time.perf_counter()
            zones = zones_from_grid(grid)
            world_from_sensor = camera @ zones.mount
            guide_ms = self._guide(zones, world_from_sensor, grid["time"])
            fused = self.volume.integrate(zones, world_from_sensor, grid["time"], camera[:3, 3])
            ended = time.perf_counter()
            self.stats["tof_frames"] += 1
            self.stats["tof_voxels"] = fused
            self.stats["fuse_ms"] = (ended - started) * 1000
            self.captured_at = grid["time"]
            # Stage times of this frame (ms): sensor to pilot (link), pilot to fusion start
            # (worker queue, waiting for the pose stream, and earlier grids of the same batch
            # being fused), fusion; _remesh adds the rest. `received` is the pilot's ito.clock,
            # which is perf_counter, system-wide.
            received = grid.get("received")
            self.timing = dict(
                link_ms=None if received is None else (received - grid["time"]) * 1000,
                queued_ms=None if received is None else (started - received) * 1000,
                integrate_ms=(ended - started) * 1000,
                guide_ms=guide_ms,
                integrated=ended,
            )
        self.grids = waiting

        news = None
        colours = now - self.last_recolour >= RECOLOUR_INTERVAL
        changed = (
            self.volume.dirty or self.volume.evicted or self.recolour
            or (colours and self.volume.painted)
        )  # fmt: skip
        if now - self.last_mesh >= MESH_INTERVAL and changed:
            self.last_mesh = now
            if colours:
                self.last_recolour = now
            self._remesh(colours)
        if self.mesh_changed or self.pending_projectors:
            self.revision += 1
            news = MeshFrame(self.revision, self.captured_at, camera=self.camera, epoch=self.epoch)
            if self.mesh_changed:
                news.positions, news.normals, news.colours, news.triangles = self.combined()
                news.timing, self.meshed_timing = self.meshed_timing, None
                self.mesh_changed = False
            news.projectors, self.pending_projectors = self.pending_projectors, []
        return news

    def _measure_frame_delay(self, now):
        """The frames' delay against the orientation track the frames are posed with: the head
        IMU's where it has one, else the pose stream's."""
        if self.frame_intrinsics is None or not self.frame_delay.due(now):
            return
        if self.head_imu and self.orientations:
            times = [t for t, _ in self.orientations]
            quaternions = [q for _, q in self.orientations]
        else:
            times = [t for t, _, _ in self.poses.samples]
            quaternions = [q for _, _, q in self.poses.samples]
        if len(times) >= 2:
            track = np.array(times), np.array(quaternions)
            self.frame_delay.estimate(now, *track, self.frame_intrinsics)
            self.stats["frame_delay_extra_ms"] = self.frame_delay.extra * 1000

    def _guide(self, zones, world_from_sensor, when):
        """Edges of this ToF frame along the camera frame nearest in time (zones.guide)."""
        if not zones_module.GUIDE or not self.posed:
            return 0.0
        exposed, image, camera, intrinsics = min(self.posed, key=lambda f: abs(f[0] - when))
        if abs(exposed - when) > GUIDE_MATCH:
            self.stats["unguided"] += 1
            return 0.0
        started = time.perf_counter()
        fx, fy, cx, cy = intrinsics
        sensor_to_image = np.linalg.inv(camera) @ world_from_sensor

        def project(points):
            p = points @ sensor_to_image[:3, :3].T + sensor_to_image[:3, 3]
            z = -p[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.where(z > 0.02, cx + fx * p[:, 0] / z, np.nan)
                v = np.where(z > 0.02, cy - fy * p[:, 1] / z, np.nan)
            return u, v

        self.stats["guided"] = zones.guide(image, project)
        self.stats["guide_dt_ms"] = (when - exposed) * 1000
        ms = (time.perf_counter() - started) * 1000
        self.stats["guide_ms"] = ms
        return ms

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
        similar = [
            (angle_between(pose[:3, :3], camera[:3, :3]), slot)
            for slot, (pose, _) in self.kept.items()
            if np.linalg.norm(pose[:3, 3] - camera[:3, 3]) < KEY_DISTANCE
        ]
        similar = [(angle, slot) for angle, slot in similar if angle < KEY_ANGLE]
        if similar:
            slot = min(similar)[1]
            if exposed - self.kept[slot][1] < KEY_REFRESH:
                return
        elif len(self.kept) < self.keyframes - 1:
            slot = self.next_kept
            self.next_kept += 1
        else:
            slot = min(self.kept, key=lambda s: self.kept[s][1])  # the oldest view
        self.kept[slot] = (camera, exposed)
        self.pending_projectors = [p for p in self.pending_projectors if p.slot != slot]
        self.pending_projectors.append(Projector(slot, image, camera, k, exposed))

    def _remesh(self, colours):
        started = time.perf_counter()
        dirty, painted, evicted = self.volume.take_dirty(colours)
        for key in evicted:
            self.meshes.pop(key, None)
        # A block's mesh reaches one voxel into its neighbours: new geometry re-meshes them too,
        # and so does a forgotten block (by its old key: its slot may hold another block now).
        todo = set(dirty.tolist())
        self.recolour.update(painted.tolist())
        self.recolour -= todo
        batch = []
        for slot in sorted(self.recolour):
            if len(batch) == RECOLOUR_BATCH:
                break
            self.recolour.discard(slot)
            if self.volume.slots.get(tuple(self.volume.keys[slot].tolist())) == slot:
                batch.append(slot)  # (an evicted slot no longer holds the repainted block)
        todo.update(batch)
        centres = [tuple(k) for k in self.volume.keys[dirty].tolist()] + list(evicted)
        for kx, ky, kz in centres:
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
        ended = time.perf_counter()
        self.stats["mesh_ms"] = (ended - started) * 1000
        if self.timing is not None:
            # The newest ToF frame is in this mesh: how long it waited for a re-mesh, and the
            # re-mesh itself (the worker adds when it was sent, the pilot when it arrived).
            self.meshed_timing = dict(
                self.timing,
                to_mesh_ms=(started - self.timing.pop("integrated")) * 1000,
                mesh_ms=(ended - started) * 1000,
                meshed_blocks=len(slots),
            )
            self.timing = None
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
