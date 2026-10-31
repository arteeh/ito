"""The ToF mesh in its own process, with the calls the pilot makes on a reconstruction worker.

The pilot thread only downsamples camera frames and queues messages; fusion (NumPy, CPU) runs in
the worker so it never holds the render loop or the link. Queues are bounded: a busy worker
drops camera frames (colour only, harmless), never motion or ToF frames (geometry), which are
small.
"""

import logging
import multiprocessing as mp
import queue
import traceback

import numpy as np

from ito import clock

log = logging.getLogger(__name__)

FRAME_RATE = 15.0  # camera frames forwarded per second
FRAME_WIDTH = 640  # their width in the worker (colour and projectors)
CLOSE_WAIT = 3.0


def _run(inputs, outputs, notes, options, integrated):
    from .fusion import Fusion, MeshFrame

    try:
        fusion = Fusion(**options)
        outbox = None
        next_note = 0.0
        while True:
            try:
                item = inputs.get(timeout=0.01)
            except queue.Empty:
                item = None
            while item is not None:
                kind = item[0]
                if kind == "stop":
                    return
                if kind == "motion":
                    fusion.motion(item[1])
                elif kind == "ranges":
                    fusion.ranges(item[1])
                elif kind == "imu":
                    fusion.imu(item[1])
                elif kind == "frame":
                    fusion.frame(*item[1:])
                try:
                    item = inputs.get_nowait()
                except queue.Empty:
                    item = None
            before = fusion.stats["tof_frames"]
            news = fusion.step(clock.now())
            integrated.value += fusion.stats["tof_frames"] - before
            if news is not None:
                outbox = news if outbox is None else _merge(outbox, news, MeshFrame)
            if outbox is not None:
                try:
                    outputs.put_nowait(outbox)
                    outbox = None
                except queue.Full:
                    pass
            now = clock.now()
            if now >= next_note:
                next_note = now + 1.0
                stats = dict(fusion.stats, volume_mb=fusion.volume.nbytes / 2**20)
                stats["capacity"] = fusion.volume.capacity
                try:
                    notes.put_nowait(("stats", stats))
                except queue.Full:
                    pass
    except Exception as exc:  # reported to the pilot, which shows the flat camera feed
        try:
            notes.put(("error", f"ToF mesh failed: {exc}", traceback.format_exc()), timeout=1)
        except queue.Full:
            pass


def _merge(old, new, frame_type):
    """Two publications as one: the newer mesh, the newest projector of every slot."""
    merged = frame_type(new.revision, new.captured_at, camera=new.camera, epoch=new.epoch)
    source = new if new.positions is not None else old
    merged.positions, merged.normals = source.positions, source.normals
    merged.colours, merged.triangles = source.colours, source.triangles
    slots = {p.slot: p for p in old.projectors}
    slots.update({p.slot: p for p in new.projectors})
    merged.projectors = list(slots.values())
    return merged


class TofMesh:
    """Pilot-side handle: submit(), motion(), ranges(), poll(), status(), statistics(), close()."""

    def __init__(self, intrinsics, *, capture_delay=0.0, voxel=0.04, capacity=6000, keyframes=12):
        context = mp.get_context("spawn")
        self.intrinsics = intrinsics
        self.capture_delay = float(capture_delay)
        self.inputs = context.Queue(maxsize=256)
        self.outputs = context.Queue(maxsize=3)
        self.notes = context.Queue(maxsize=16)
        self.integrated = context.Value("Q", 0, lock=False)
        options = dict(voxel=voxel, capacity=capacity, keyframes=keyframes)
        self.process = context.Process(
            target=_run,
            args=(self.inputs, self.outputs, self.notes, options, self.integrated),
            name="ito-tofmesh",
            daemon=True,
        )
        self.process.start()
        self.started = clock.now()
        self.last_frame = -1e9
        self.stats = {}
        self.error = None
        self.has_mesh = False
        self._max_splats = 1
        self._closed = False
        # Frames go to the worker strided to about FRAME_WIDTH: pixel i of the strided image is
        # pixel i * step + step // 2 of the full one.
        step = self.step = max(1, round(intrinsics.width / FRAME_WIDTH))
        self.frame_intrinsics = (
            intrinsics.fx / step,
            intrinsics.fy / step,
            (intrinsics.cx - step // 2) / step,
            (intrinsics.cy - step // 2) / step,
        )

    # -- inputs -------------------------------------------------------------------------
    def _put(self, item, block=False):
        try:
            self.inputs.put(item, timeout=0.05) if block else self.inputs.put_nowait(item)
            return True
        except queue.Full:
            return False

    def submit(self, rgb, depth=None, camera=None, captured_at=None):
        """A camera frame (h, w, 3) uint8 at the described pinhole; `camera` is the driver's
        pose for it (used only when the motion stream does not cover the frame)."""
        captured_at = clock.now() if captured_at is None else captured_at
        if captured_at - self.last_frame < 1 / FRAME_RATE:
            return False
        self.last_frame = captured_at
        s = self.step
        image = np.ascontiguousarray(rgb[s // 2 :: s, s // 2 :: s]) if s > 1 else rgb
        intrinsics = self.frame_intrinsics
        fallback = None if camera is None else np.asarray(camera, np.float64)
        return self._put(("frame", image, captured_at - self.capture_delay, intrinsics, fallback))

    def motion(self, samples):
        """Driver motion samples: [(pilot-clock time, 4x4 world_from_camera, ...), ...]."""
        if samples:
            self._put(("motion", [(s[0], np.asarray(s[1], np.float64)) for s in samples]), True)

    def ranges(self, grids):
        if grids:
            self._put(("ranges", grids), True)

    def imu(self, samples):
        """Head IMU: [(pilot-clock time, xyzw sensor-to-world, 3x3 sensor-to-camera), ...]."""
        if samples:
            self._put(("imu", samples), True)

    # -- outputs ------------------------------------------------------------------------
    def _notes(self):
        while True:
            try:
                note = self.notes.get_nowait()
            except queue.Empty:
                return
            if note[0] == "stats":
                self.stats = note[1]
            elif note[0] == "error":
                self.error = note[1]
                raise RuntimeError(f"{note[1]}\n{note[2]}")

    def poll(self):
        self._notes()
        if not self.process.is_alive() and not self._closed:
            raise RuntimeError("ToF mesh worker stopped")
        try:
            frame = self.outputs.get_nowait()
        except queue.Empty:
            return None
        if frame.positions is not None and len(frame.triangles):
            self.has_mesh = True
        return frame

    def status(self):
        if self.error:
            return self.error
        if not self.stats.get("tof_frames"):
            if clock.now() - self.started > 5:
                return "ToF mesh: waiting for ToF frames (does the robot stream its ToF?)"
            return "Starting ToF mesh"
        s = self.stats
        return f"ToF mesh: {s['triangles']:,} triangles, {s['blocks']:,}/{s['capacity']:,} blocks"

    def statistics(self):
        s = self.stats
        return {
            "tofmesh_triangles": s.get("triangles"),
            "tofmesh_blocks": s.get("blocks"),
            "tofmesh_fuse_ms": s.get("fuse_ms"),
            "tofmesh_mesh_ms": s.get("mesh_ms"),
            "tofmesh_tof_frames": s.get("tof_frames"),
            "tofmesh_tof_dropped": s.get("tof_dropped"),
            "tofmesh_volume_mb": s.get("volume_mb"),
        }

    def pose(self):
        return None

    def display_completed(self, began, period):
        pass

    @property
    def max_splats(self):
        return self._max_splats

    def set_max_splats(self, value):
        self._max_splats = value

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._put(("stop",), True)
        self.process.join(CLOSE_WAIT)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1)
        for q in (self.inputs, self.outputs, self.notes):
            q.cancel_join_thread()
            q.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
