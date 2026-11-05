"""The ToF mesh in its own process, with the calls the pilot makes on it.

The pilot thread only downsamples camera frames and queues messages; fusion (NumPy, CPU) runs in
the worker so it never holds the render loop or the link. Queues are bounded and nothing here
ever blocks the link loop: a busy worker drops camera frames (colour only, harmless); motion,
ToF and IMU batches (geometry, small) wait in a short backlog, and only a worker stalled for
seconds loses some of them (counted in the statistics and the status line).

With Depth Anything 3 chosen, its model runs in a third process (pda.py), started here and
fed by the worker: one camera frame and prompt at a time, the next once its depth map is in.
"""

import logging
import multiprocessing as mp
import queue
import traceback
from collections import deque

import numpy as np

from ito import clock, diagnostics

from . import pda

log = logging.getLogger(__name__)

FRAME_RATE = 15.0  # camera frames forwarded per second
FRAME_WIDTH = 640  # their width in the worker (colour and projectors)
# Stopping the worker: STOP_WAIT to queue the stop, EXIT_WAIT to leave its loop and exit, then
# KILL_WAIT after a terminate. CLOSE_WAIT is all of it: what close() may take.
STOP_WAIT, EXIT_WAIT, KILL_WAIT = 0.5, 3.0, 1.0
CLOSE_WAIT = STOP_WAIT + EXIT_WAIT + KILL_WAIT
BACKLOG = 64  # geometry batches held while the worker's queue is full (a few seconds)
TIMING_SUMMARY = 1.0  # seconds of mesh stage times per diagnostic log line


def frame_size(width, height):
    """The size of the camera frames the worker takes, strided to about FRAME_WIDTH from the
    (width, height) frame, and the stride: their pixel i is pixel i * step + step // 2."""
    step = max(1, round(width / FRAME_WIDTH))
    return (len(range(step // 2, width, step)), len(range(step // 2, height, step))), step


def _run(inputs, outputs, notes, options, integrated, depth):
    from .fusion import Fusion, MeshFrame

    try:
        fusion = Fusion(**options)
        outbox = None
        next_note = 0.0
        waiting = False  # a frame is with Depth Anything 3 (depth: its two queues)
        maps = 0  # depth maps fused up to the last note
        while True:
            try:
                item = inputs.get(timeout=0.01)
            except queue.Empty:
                item = None
            while item is not None:
                kind = item[0]
                if kind == "stop":
                    # The pilot reads nothing more: what is still queued for it is dropped.
                    # Otherwise the process would not exit until the queues' feeder threads
                    # had written it all to pipes nobody reads, and close() would wait out
                    # EXIT_WAIT and terminate it (3 s with the app frozen).
                    outputs.cancel_join_thread()
                    notes.cancel_join_thread()
                    return
                if kind == "motion":
                    fusion.motion(item[1])
                elif kind == "ranges":
                    fusion.ranges(item[1])
                elif kind == "imu":
                    fusion.imu(item[1])
                elif kind == "frame":
                    fusion.frame(*item[1:])
                elif kind == "depth_failed":
                    fusion.depth_failed(item[1])
                    depth = None
                try:
                    item = inputs.get_nowait()
                except queue.Empty:
                    item = None
            if depth is not None:
                waiting, depth = _depth(fusion, depth, waiting)
            before = fusion.stats["tof_frames"]
            news = fusion.step(clock.now())
            integrated.value += fusion.stats["tof_frames"] - before
            if news is not None:
                outbox = news if outbox is None else _merge(outbox, news, MeshFrame)
            if outbox is not None:
                if outbox.timing is not None:
                    outbox.timing["sent"] = clock.now()
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
                stats["pda_rate"], maps = fusion.stats["pda_maps"] - maps, fusion.stats["pda_maps"]
                try:
                    notes.put_nowait(("stats", stats))
                except queue.Full:
                    pass
    except Exception as exc:  # reported to the pilot, which shows the flat camera feed
        try:
            notes.put(("error", f"ToF mesh failed: {exc}", traceback.format_exc()), timeout=1)
        except queue.Full:
            pass


def _depth(fusion, queues, waiting):
    """Take Depth Anything 3's answer if one is in and send it the next frame: (waiting,
    queues), queues None once it failed."""
    requests, results = queues
    try:
        answer = results.get_nowait()
    except queue.Empty:
        answer = None
    if answer is not None:
        if answer[0] == "depth":
            fusion.depth_map(*answer[1:])
            waiting = False
        elif answer[0] == "error":
            fusion.depth_failed(answer[1])
            return False, None
    if not waiting:
        request = fusion.depth_request()
        if request is not None:
            try:
                requests.put_nowait(request)
                waiting = True
            except queue.Full:
                pass
    return waiting, queues


def _merge(old, new, frame_type):
    """Two publications as one: the newer mesh, the newest projector of every slot."""
    source = new if new.positions is not None else old
    # The capture time goes with the mesh it timed (the HUD's capture-to-visible).
    merged = frame_type(new.revision, source.captured_at, camera=new.camera, epoch=new.epoch)
    merged.positions, merged.normals = source.positions, source.normals
    merged.colours, merged.triangles = source.colours, source.triangles
    merged.blocks = source.blocks
    merged.cleared = sorted(set(old.cleared) | set(new.cleared))
    merged.timing = source.timing
    merged.guess = new.guess
    slots = {p.slot: p for p in old.projectors}
    slots.update({p.slot: p for p in new.projectors})
    merged.projectors = list(slots.values())
    return merged


class TofMesh:
    """Pilot-side handle: submit(), motion(), ranges(), poll(), status(), statistics(), close()."""

    def __init__(
        self, intrinsics, memory, *, capture_delay=0.0, voxel=0.04, capacity=6000, depth=None
    ):
        """depth: Depth Anything 3's settings, dict(model, inside, outside), or None for
        the renderer's guessed depth (also what a missing model or a failure falls back to)."""
        context = mp.get_context("spawn")
        self.intrinsics = intrinsics
        self.capture_delay = float(capture_delay)
        self.inputs = context.Queue(maxsize=256)
        self.outputs = context.Queue(maxsize=3)
        self.notes = context.Queue(maxsize=16)
        self.integrated = context.Value("Q", 0, lock=False)
        # A camera whose frames carry their own capture time (capture_delay 0) needs no measured
        # delay; the estimate is the fallback for one that only guesses (latency.FrameDelay).
        options = dict(
            memory=memory, voxel=voxel, capacity=capacity, frame_delay=self.capture_delay > 0
        )
        self.depth_error = None
        self.pda = queues = None
        if depth is not None:
            self.depth_error = pda.missing(depth["model"])
            if self.depth_error:
                log.error("%s; using guessed depth", self.depth_error)
            else:
                queues = (context.Queue(maxsize=1), context.Queue(maxsize=4))
                self.pda = context.Process(
                    target=pda.run, args=(*queues, depth["model"]), name="ito-pda", daemon=True
                )
                self.pda.start()
                options["depth"] = dict(inside=depth["inside"], outside=depth["outside"])
                # The depth maps fill the whole camera view, not the ToF's ~45 degrees of it.
                options["capacity"] = 2 * capacity
                options["max_range"] = 12.0
                options["voxel"] = 0.04
        self.process = context.Process(
            target=_run,
            args=(self.inputs, self.outputs, self.notes, options, self.integrated, queues),
            name="ito-tofmesh",
            daemon=True,
        )
        self.process.start()
        self.started = clock.now()
        self.last_frame = -1e9
        self.stats = {}
        self.error = None
        self.has_mesh = False
        self._closed = False
        self.backlog = deque()
        self.dropped = 0
        self.timings = []  # this second's mesh stage times, for the diagnostic log
        self.timings_since = clock.now()
        _, step = frame_size(intrinsics.width, intrinsics.height)
        self.frame_intrinsics = (
            intrinsics.fx / step,
            intrinsics.fy / step,
            (intrinsics.cx - step // 2) / step,
            (intrinsics.cy - step // 2) / step,
        )

    # -- inputs -------------------------------------------------------------------------
    def _put(self, item, keep=False):
        """Queue without blocking. `keep` (geometry): if the queue is full, hold the item in the
        backlog, in order, dropping the oldest past BACKLOG."""
        if self._closed:
            return False
        try:
            while self.backlog:
                self.inputs.put_nowait(self.backlog[0])
                self.backlog.popleft()
            self.inputs.put_nowait(item)
            return True
        except queue.Full:
            if keep:
                self.backlog.append(item)
                while len(self.backlog) > BACKLOG:
                    self.backlog.popleft()
                    self.dropped += 1
            return False
        except ValueError:  # the queue was closed under us (the pilot retiring this worker)
            return False

    def submit(self, image, camera, captured_at):
        """A camera frame at the described pinhole, strided as frame_size says, (h, w, 3) uint8;
        `camera` is the driver's pose for it (used only when the motion stream does not
        cover the frame)."""
        if captured_at - self.last_frame < 1 / FRAME_RATE:
            return False
        self.last_frame = captured_at
        intrinsics = self.frame_intrinsics
        fallback = None if camera is None else np.asarray(camera, np.float64)
        return self._put(("frame", image, captured_at - self.capture_delay, intrinsics, fallback))

    def motion(self, samples):
        """Driver motion samples: [(pilot-clock time, 4x4 world_from_camera, ...), ...]."""
        if samples:
            self._put(
                ("motion", [(s[0], np.asarray(s[1], np.float64)) for s in samples]), keep=True
            )

    def ranges(self, grids):
        if grids:
            self._put(("ranges", grids), keep=True)

    def imu(self, samples):
        """Head IMU: [(pilot-clock time, xyzw sensor-to-world, 3x3 sensor-to-camera, gyro), ...]."""
        if samples:
            self._put(("imu", samples), keep=True)

    # -- outputs ------------------------------------------------------------------------
    def _notes(self):
        while True:
            try:
                note = self.notes.get_nowait()
            except queue.Empty:
                return
            if note[0] == "stats":
                self.stats = note[1]
                if self.stats.get("pda_error") and not self.depth_error:
                    self.depth_error = self.stats["pda_error"]
                    log.error("%s; using guessed depth", self.depth_error)
            elif note[0] == "error":
                self.error = note[1]
                raise RuntimeError(f"{note[1]}\n{note[2]}")

    def poll(self):
        self._notes()
        if not self.process.is_alive() and not self._closed:
            raise RuntimeError("ToF mesh worker stopped")
        if self.pda is not None and not self.pda.is_alive() and not self._closed:
            self.pda = None
            if not self.depth_error:
                self._put(("depth_failed", "Depth Anything 3 stopped"), keep=True)
        try:
            frame = self.outputs.get_nowait()
        except queue.Empty:
            return None
        if frame.positions is not None and len(frame.triangles):
            self.has_mesh = True
        if frame.timing is not None and frame.captured_at is not None:
            self._timing(frame)
        return frame

    def _timing(self, frame):
        """Where the time went between a ToF frame and its mesh reaching the pilot (the windows
        add the draw). Meshes arrive ~12.7 a second, so the diagnostic log (ITO_DEBUG /
        diagnostic logging only) gets one summary a second: count, median and max per stage."""
        now = clock.now()
        row = {k: v for k, v in frame.timing.items() if k != "sent" and v is not None}
        row["sensor_to_pilot_ms"] = (now - frame.captured_at) * 1000
        if frame.timing.get("sent") is not None:
            row["to_pilot_ms"] = (now - frame.timing["sent"]) * 1000
        row["triangles"] = len(frame.triangles)
        self.timings.append(row)
        if now - self.timings_since < TIMING_SUMMARY:
            return
        rows, self.timings, self.timings_since = self.timings, [], now
        summary = {}
        for key in sorted({k for r in rows for k in r}):
            values = [r[key] for r in rows if key in r]
            summary[key] = float(np.median(values))
            summary[key + "_max"] = float(max(values))
        diagnostics.event("tofmesh_mesh", meshes=len(rows), **summary)

    def status(self):
        if self.error:
            return self.error
        if not self.stats.get("tof_frames"):
            if clock.now() - self.started > 5:
                return "ToF mesh: waiting for ToF frames (does the robot stream its ToF?)"
            return "Starting ToF mesh"
        s = self.stats
        text = f"ToF mesh: {s['triangles']:,} triangles, {s['blocks']:,}/{s['capacity']:,} blocks"
        if self.dropped:
            text += f"; worker busy, {self.dropped:,} sensor batches lost"
        if self.depth_error:
            text += f"; {self.depth_error}, using guessed depth"
        elif self.pda is not None:
            text += f"; Depth Anything 3 {s.get('pda_rate', 0)} maps/s"
        return text

    def statistics(self):
        s = self.stats
        return {
            "tofmesh_triangles": s.get("triangles"),
            "tofmesh_blocks": s.get("blocks"),
            "tofmesh_fuse_ms": s.get("fuse_ms"),
            "tofmesh_mesh_ms": s.get("mesh_ms"),
            "tofmesh_tof_frames": s.get("tof_frames"),
            "tofmesh_tof_dropped": s.get("tof_dropped"),
            "tofmesh_frame_delay_extra_ms": s.get("frame_delay_extra_ms"),
            "tofmesh_frame_delay_moves": s.get("frame_delay_moves"),
            "tofmesh_untilted": s.get("untilted"),
            "tofmesh_tilt_held": s.get("tilt_held"),
            "tofmesh_tilt_ahead": s.get("tilt_ahead"),
            "tofmesh_volume_mb": s.get("volume_mb"),
            "tofmesh_ingress_dropped": self.dropped,
            "pda_maps": s.get("pda_maps"),
            "pda_ms": s.get("pda_ms"),
            "pda_fuse_ms": s.get("pda_fuse_ms"),
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.inputs.put(("stop",), timeout=STOP_WAIT)
        except (queue.Full, ValueError):
            pass
        if self.pda is not None:
            self.pda.terminate()  # nothing to save in it
        self.process.join(EXIT_WAIT)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(KILL_WAIT)
        if self.pda is not None:
            self.pda.join(KILL_WAIT)
        for q in (self.inputs, self.outputs, self.notes):
            q.cancel_join_thread()
            q.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
