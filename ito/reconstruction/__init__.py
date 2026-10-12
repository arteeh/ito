"""Nonblocking RGB-D/monocular input and incremental shared-memory scene output."""

import json
import logging
import multiprocessing as mp
import os
import time

import numpy as np

from ito import clock
from ito.protocol import Intrinsics
from ito.render.pose import validate_pose

from .ring import MotionRing, SplatUpdate, UpdateRing
from .slam_vision import RESTART_CAUSES

__all__ = ["Reconstruction", "SplatUpdate", "default_budget"]

log = logging.getLogger(__name__)
# A worker gets this long to see the stop flag; one busy in a step that never ends is killed.
STOP_WAIT = 0.5


def default_budget(renderer: str = "", video_memory_mb: int = 0) -> int:
    if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
        return 16_384
    return 400_000 if video_memory_mb >= 8192 else 262_144


def _run(recon):
    try:
        robot_first = recon.backend == "slam" and recon.tracking_mode == "robot"
        if robot_first:
            try:
                from .mapped_slam import MappedSLAMBackend as SLAMBackend
            except ImportError as exc:
                raise RuntimeError("Robot-first tracking needs Ito's SLAM components") from exc

            backend = SLAMBackend(
                recon.budget.value,
                recon.intrinsics,
                report=recon.report,
                motion_noise=recon.motion_noise,
                frame_delay=recon.frame_delay,
                exposure=recon.exposure,
                **recon.options,
            )
        elif recon.backend == "slam":
            from .slam_vision import VisionSLAMBackend

            backend = VisionSLAMBackend(
                recon.budget.value,
                recon.intrinsics,
                report=recon.report,
                **recon.options,
            )
        else:
            from .mapped_rgbd import MappedRGBDBackend

            backend = MappedRGBDBackend(
                recon.budget.value, recon.intrinsics, report=recon.report, **recon.options
            )
        scene = getattr(backend, "scene", backend)
        sequence = 0
        # ITO_TIMELINE names a file for one line per frame the worker took: what became of it.
        timeline = os.environ.get("ITO_TIMELINE")
        timeline = open(timeline, "a", buffering=1) if timeline else None
        cursor = 0
        motion_cursor = 0
        revision = 0
        captured = recon.epoch
        # A pilot app that died without closing it would otherwise leave it holding the GPU.
        parent = mp.parent_process()
        while not recon.stopped.value and parent.is_alive():
            now = clock.now() - recon.epoch
            scene.expire(now, recon.ring.last_acknowledged())
            if backend.budget != recon.budget.value:
                backend.resize(recon.budget.value, now)
                if scene is not backend:
                    scene.resize(recon.budget.value, now)
            if robot_first:
                rows, motion_cursor = recon.motion_ring.read(motion_cursor)
                backend.add_motion(rows)
            frame = None
            if recon.input_lock.acquire(False):
                try:
                    if recon.sequence.value != sequence:
                        sequence = recon.sequence.value
                        frame = (
                            np.frombuffer(recon.rgb, np.uint8).reshape(recon.shape + (3,)).copy(),
                            np.frombuffer(recon.depth, np.float32).reshape(recon.shape).copy(),
                            np.frombuffer(recon.camera, np.float32).reshape(4, 4).copy(),
                        )
                        captured = recon.captured.value
                        measured = bool(recon.measured.value)
                finally:
                    recon.input_lock.release()
            if frame is not None:
                if timeline:
                    before = (
                        getattr(backend, "tracked", 0),
                        sum(getattr(backend, "restarts", {}).values()),
                        scene.count,
                    )
                    began = clock.now()
                if robot_first:
                    backend.integrate(*frame, now, captured=captured)
                elif recon.backend == "slam":
                    backend.integrate(*frame, now, measured=measured)
                else:
                    backend.integrate(*frame, now, captured=captured)
                recon.integrated.value += 1
                if timeline:
                    done = clock.now()
                    restarted = sum(getattr(backend, "restarts", {}).values()) - before[1]
                    tracked = getattr(backend, "tracked", 0) - before[0]
                    since = getattr(backend, "unmatched_since", None)
                    row = dict(
                        seq=sequence,
                        captured=round(captured, 4),
                        began=round(began, 4),
                        age_ms=round((began - captured) * 1000, 1),
                        work_ms=round((done - began) * 1000, 1),
                        measured=measured if recon.backend == "slam" else None,
                        outcome="integrated"
                        if recon.backend != "slam"
                        else "restart"
                        if restarted
                        else "tracked"
                        if tracked
                        else "lost"
                        if getattr(backend, "lost", False)
                        else "unmatched"
                        if since is not None
                        else "skipped",
                        splats=int(scene.count),
                        splats_delta=int(scene.count) - before[2],
                        recalls=getattr(backend, "recalls", None),
                        matched=getattr(backend, "matched", None),
                        blurred=getattr(backend, "blurred", None),
                        frame_delay_ms=round(backend.graph.td * 1000, 1)
                        if hasattr(backend, "graph")
                        else None,
                    )
                    timeline.write(json.dumps(row) + "\n")
                if recon.backend == "slam" and recon.output_lock.acquire(False):
                    try:
                        np.frombuffer(recon.output_camera, np.float32)[:] = (
                            backend.camera_pose.ravel()
                        )
                        np.frombuffer(recon.output_correction, np.float32)[:] = (
                            backend.correction.ravel() if robot_first else np.eye(4).ravel()
                        )
                        recon.output_captured.value = captured
                        recon.tracked.value = backend.tracked
                        recon.tracking.value = not backend.lost and backend.tracked > 0
                        recon.restart_counts[:] = [backend.restarts[c] for c in RESTART_CAUSES]
                    finally:
                        recon.output_lock.release()
            # Bound work per turn; dirty slots coalesce while the display is behind.
            for _ in range(recon.ring.slots):
                dirty = np.flatnonzero(scene.dirty)
                start = np.searchsorted(dirty, cursor)
                indices = np.concatenate((dirty[start:], dirty[:start]))[: recon.ring.batch]
                if not len(indices):
                    break
                if not recon.ring.publish(
                    indices,
                    scene.records[indices],
                    len(scene.keys),
                    scene.count,
                    revision,
                    getattr(scene, "captured_at", captured),
                    getattr(scene, "antialiased", False),
                ):
                    break
                retiring = indices[scene.retiring[indices]]
                scene.retire_revision[retiring] = revision
                cursor = (int(indices[-1]) + 1) % len(scene.keys)
                scene.dirty[indices] = False
                revision += 1
            time.sleep(0.01)
    except Exception as exc:
        # The pilot sees one plain line; the log keeps the cause for whoever debugs it.
        log.exception("Reconstruction stopped")
        message = str(exc) if recon.backend == "slam" else f"Reconstruction failed: {exc}"
        recon.errors.send(message[:2000])
    finally:
        if "backend" in locals() and hasattr(backend, "close"):
            backend.close()
        recon.errors.close()


class Reconstruction:
    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        max_splats=262_144,
        voxel_size=0.04,
        fade_seconds=0.5,
        device="auto",
        backend="rgbd",
        tracking="robot",
        motion_noise=None,
        frame_delay=0.0,
        exposure=None,
    ):
        """tracking: for SLAM, "robot" fuses the robot's Motion with vision (it needs the
        robot's motion_noise); "vision" is MASt3R-SLAM alone, steered by the pose prior
        each frame carries."""
        if not 1 <= max_splats <= 4_194_304:
            raise ValueError("Max splats must be between 1 and 4,194,304")
        if not all(np.isfinite(v) and v > 0 for v in (voxel_size, fade_seconds)):
            raise ValueError("Voxel size and fade must be finite and positive")
        if backend not in ("rgbd", "slam"):
            raise ValueError("Reconstruction backend must be rgbd or slam")
        if tracking not in ("robot", "vision"):
            raise ValueError("Tracking must be robot or vision")
        self.backend = backend
        self.tracking_mode = tracking if motion_noise is not None else "vision"
        self.motion_noise = motion_noise
        self.frame_delay = frame_delay
        self.exposure = exposure
        self.intrinsics = intrinsics
        self.shape = (intrinsics.height, intrinsics.width)
        self.options = dict(
            voxel_size=voxel_size,
            fade_seconds=fade_seconds,
            device=device,
        )
        self.epoch = clock.now()
        context = mp.get_context("spawn")
        self.status_lock = context.Lock()
        self.status_text = context.RawArray("B", 1024)
        self.output_lock = context.Lock()
        self.output_camera = context.RawArray("f", 16)
        self.output_correction = context.RawArray("f", np.eye(4).ravel().tolist())
        self.motion_ring = MotionRing(context)
        self.output_captured = context.RawValue("d", 0)
        self.tracked = context.RawValue("Q", 0)
        self.integrated = context.RawValue("Q", 0)  # frames integrated into the scene
        self.tracking = context.RawValue("b", False)
        self.restart_counts = context.RawArray("Q", len(RESTART_CAUSES))
        self.report(
            "Posed RGB-D"
            if backend != "slam"
            else "Starting robot-first tracking"
            if self.tracking_mode == "robot"
            else "Starting MASt3R-SLAM"
        )
        # The worker publishes only between integrations, which take ~100 ms with
        # SLAM; a whole frame's refreshed slots must fit in one turn or they queue
        # behind the next frame. The display still drains a bounded few per frame.
        self.ring = UpdateRing(context, self.epoch, fade_seconds, slots=16)
        pixels = intrinsics.width * intrinsics.height
        self.rgb = context.RawArray("B", pixels * 3)
        self.depth = context.RawArray("f", pixels)
        self.camera = context.RawArray("f", 16)
        self.captured = context.RawValue("d", 0)
        self.measured = context.RawValue("b", False)
        self.sequence = context.RawValue("Q", 0)
        self.budget = context.RawValue("I", max_splats)
        self.input_lock = context.Lock()
        self.stopped = context.RawValue("b", False)
        reader, self.errors = context.Pipe(duplex=False)
        self.process = None
        process = context.Process(target=_run, args=(self,), name="ito-reconstruction")
        process.start()
        self.process = process
        self.errors.close()
        self.errors = reader
        self.closed = False

    @property
    def max_splats(self):
        return self.budget.value

    def set_max_splats(self, value):
        if not 1 <= value <= 4_194_304:
            raise ValueError("Max splats must be between 1 and 4,194,304")
        self.budget.value = value

    def report(self, message):
        if self.status_lock.acquire(False):
            try:
                encoded = message.encode("utf-8")[:1023]
                self.status_text[: len(encoded)] = encoded
                self.status_text[len(encoded)] = 0
            finally:
                self.status_lock.release()

    def status(self):
        if not self.status_lock.acquire(False):
            return None
        try:
            return bytes(self.status_text).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        finally:
            self.status_lock.release()

    def pose(self):
        if not self.output_lock.acquire(False):
            return None
        try:
            return (
                np.frombuffer(self.output_camera, np.float32).reshape(4, 4).copy(),
                self.tracked.value,
                bool(self.tracking.value),
                self.output_captured.value,
            )
        finally:
            self.output_lock.release()

    def correction(self):
        """World-from-odometry for the newest pose: the robot's own camera pose, moved by it,
        is the fused one. Identity unless tracking is robot-first; None while publishing."""
        if not self.output_lock.acquire(False):
            return None
        try:
            return np.frombuffer(self.output_correction, np.float32).reshape(4, 4).copy()
        finally:
            self.output_lock.release()

    def motion(self, samples):
        """The robot's camera motion on the pilot clock, (time, world-from-camera, rad/s,
        state) oldest first. Never waits: a busy worker gets them with the next batch."""
        if self.tracking_mode == "robot" and self.backend == "slam":
            self.motion_ring.put(samples)

    def restarts(self):
        """SLAM's local map restarts so far by cause; None while the worker is publishing."""
        if not self.output_lock.acquire(False):
            return None
        try:
            return dict(zip(RESTART_CAUSES, self.restart_counts, strict=True))
        finally:
            self.output_lock.release()

    def submit(self, rgb, depth=None, camera=None, captured_at=None, *, measured=False):
        """Depth is axial metres; pose is world-from-camera (+Y up, -Z forward).

        SLAM needs no depth, and takes the pose only as the robot's own estimate: its
        orientation aligns the map to gravity and the startup heading, and, when measured
        (the robot reports its body heading and head pan), keeps the map's heading honest.
        RGB and depth must be synchronized and registered to these intrinsics.
        Capture time must already be corrected to the pilot monotonic clock.
        A busy mailbox drops the incoming frame; the caller never waits.
        """
        if self.closed:
            raise RuntimeError("Reconstruction is closed")
        if rgb.shape != self.shape + (3,) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 HxWx3 matching the camera intrinsics")
        if self.backend == "rgbd":
            if (
                camera is None
                or depth is None
                or depth.shape != self.shape
                or depth.dtype != np.float32
            ):
                raise ValueError("RGB-D needs float32 HxW depth and camera pose from the driver")
        camera = np.eye(4, dtype=np.float32) if camera is None else validate_pose(camera)
        captured_at = clock.now() if captured_at is None else captured_at
        if not np.isfinite(captured_at):
            raise ValueError("Capture time must be finite")
        if not self.input_lock.acquire(False):
            return False
        try:
            if self.sequence.value and captured_at <= self.captured.value:
                return False
            np.frombuffer(self.rgb, np.uint8)[:] = rgb.ravel()
            np.frombuffer(self.camera, np.float32)[:] = camera.ravel()
            if self.backend == "rgbd":
                np.frombuffer(self.depth, np.float32)[:] = depth.ravel()
            self.captured.value = captured_at
            self.measured.value = measured
            self.sequence.value += 1
            return True
        finally:
            self.input_lock.release()

    def poll(self):
        if self.closed:
            return None
        try:
            message = self.errors.recv() if self.errors.poll() else None
        except (EOFError, OSError):
            # A dead writer reads as EOF on POSIX and as a broken pipe on Windows.
            message = "Reconstruction process exited unexpectedly"
        if message is not None:
            raise RuntimeError(message)
        if not self.process.is_alive():
            raise RuntimeError(f"Reconstruction process exited ({self.process.exitcode})")
        return self.ring.poll()

    def close(self):
        """Bounded even for a suspended or wedged worker."""
        if not self.closed:
            self.closed = True
            # A killed worker may leave an Event's internal condition locked forever.
            self.stopped.value = True
            self.process.join(timeout=STOP_WAIT)
            if self.process.is_alive():
                # SIGTERM stays pending on a stopped process; SIGKILL does not.
                log.warning(
                    "Reconstruction process %d killed: still busy %.1f s after stop",
                    self.process.pid,
                    STOP_WAIT,
                )
                self.process.kill()
                self.process.join(timeout=3)
                if self.process.is_alive():
                    log.error("Reconstruction process %d did not exit", self.process.pid)
            self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
