"""Nonblocking RGB-D/monocular input and incremental shared-memory scene output."""

import multiprocessing as mp
import time

import numpy as np

from ito.protocol import Intrinsics
from ito.render.pose import validate_pose

from .rgbd import RGBDBackend
from .ring import SplatUpdate, UpdateRing

__all__ = ["Reconstruction", "SplatUpdate", "default_budget"]


def default_budget(renderer: str = "", video_memory_mb: int = 0) -> int:
    if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
        return 16_384
    return 1_048_576 if video_memory_mb >= 8192 else 262_144


def _run(recon):
    try:
        if recon.backend == "slam":
            from .slam import SLAMBackend

            backend = SLAMBackend(
                recon.budget.value,
                recon.intrinsics,
                report=recon.report,
                origin=recon.origin,
                **recon.options,
            )
        else:
            backend = RGBDBackend(recon.budget.value, recon.intrinsics, **recon.options)
        sequence = 0
        cursor = 0
        revision = 0
        captured = recon.epoch
        while not recon.stopped.value:
            now = time.monotonic() - recon.epoch
            backend.expire(now, recon.ring.last_acknowledged())
            if backend.budget != recon.budget.value:
                backend.resize(recon.budget.value, now)
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
                finally:
                    recon.input_lock.release()
            if frame is not None:
                backend.integrate(*frame, now)
                if recon.backend == "slam" and recon.output_lock.acquire(False):
                    try:
                        np.frombuffer(recon.output_camera, np.float32)[:] = (
                            backend.camera_pose.ravel()
                        )
                        recon.tracked.value = backend.tracked
                        recon.tracking.value = not backend.lost and backend.tracked > 0
                    finally:
                        recon.output_lock.release()
            # Bound work per turn; dirty slots coalesce while the display is behind.
            for _ in range(recon.ring.slots):
                dirty = np.flatnonzero(backend.dirty)
                start = np.searchsorted(dirty, cursor)
                indices = np.concatenate((dirty[start:], dirty[:start]))[: recon.ring.batch]
                if not len(indices):
                    break
                if not recon.ring.publish(
                    indices,
                    backend.records[indices],
                    len(backend.keys),
                    backend.count,
                    revision,
                    captured,
                ):
                    break
                retiring = indices[backend.retiring[indices]]
                backend.retire_revision[retiring] = revision
                cursor = (int(indices[-1]) + 1) % len(backend.keys)
                backend.dirty[indices] = False
                revision += 1
            time.sleep(0.01)
    except Exception as exc:
        message = str(exc) if recon.backend == "slam" else f"Reconstruction failed: {exc}"
        recon.errors.send(message[:2000])
    finally:
        recon.errors.close()


class Reconstruction:
    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        max_splats=262_144,
        voxel_size=0.04,
        window_seconds=4.0,
        fade_seconds=0.5,
        device="auto",
        backend="rgbd",
        origin=None,
    ):
        if not 1 <= max_splats <= 4_194_304:
            raise ValueError("Max splats must be between 1 and 4,194,304")
        if not all(np.isfinite(v) and v > 0 for v in (voxel_size, window_seconds, fade_seconds)):
            raise ValueError("Voxel size, temporal window and fade must be finite and positive")
        if backend not in ("rgbd", "slam"):
            raise ValueError("Reconstruction backend must be rgbd or slam")
        self.backend = backend
        self.origin = np.eye(4, dtype=np.float32) if origin is None else validate_pose(origin)
        self.intrinsics = intrinsics
        self.shape = (intrinsics.height, intrinsics.width)
        self.options = dict(
            voxel_size=voxel_size,
            window_seconds=window_seconds,
            fade_seconds=fade_seconds,
            device=device,
        )
        self.epoch = time.monotonic()
        context = mp.get_context("spawn")
        self.status_lock = context.Lock()
        self.status_text = context.RawArray("B", 1024)
        self.output_lock = context.Lock()
        self.output_camera = context.RawArray("f", 16)
        self.tracked = context.RawValue("Q", 0)
        self.tracking = context.RawValue("b", False)
        self.report("Starting MASt3R-SLAM" if backend == "slam" else "Posed RGB-D")
        self.ring = UpdateRing(context, self.epoch, fade_seconds)
        pixels = intrinsics.width * intrinsics.height
        self.rgb = context.RawArray("B", pixels * 3)
        self.depth = context.RawArray("f", pixels)
        self.camera = context.RawArray("f", 16)
        self.captured = context.RawValue("d", 0)
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
            )
        finally:
            self.output_lock.release()

    def submit(self, rgb, depth=None, camera=None, captured_at=None):
        """Depth is axial metres; pose is world-from-camera (+Y up, -Z forward).

        RGB and depth must be synchronized and registered to these intrinsics.
        Capture time must already be corrected to the pilot monotonic clock.
        A busy mailbox drops the incoming frame; the caller never waits.
        """
        if self.closed:
            raise RuntimeError("Reconstruction is closed")
        if rgb.shape != self.shape + (3,) or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 HxWx3 matching the camera intrinsics")
        if self.backend == "rgbd":
            if depth is None or depth.shape != self.shape or depth.dtype != np.float32:
                raise ValueError("RGB-D needs float32 HxW depth and camera pose from the driver")
            camera = validate_pose(camera)
        captured_at = time.monotonic() if captured_at is None else captured_at
        if not np.isfinite(captured_at):
            raise ValueError("Capture time must be finite")
        if not self.input_lock.acquire(False):
            return False
        try:
            np.frombuffer(self.rgb, np.uint8)[:] = rgb.ravel()
            if self.backend == "rgbd":
                np.frombuffer(self.depth, np.float32)[:] = depth.ravel()
                np.frombuffer(self.camera, np.float32)[:] = camera.ravel()
            self.captured.value = captured_at
            self.sequence.value += 1
            return True
        finally:
            self.input_lock.release()

    def poll(self):
        if self.closed:
            return None
        if self.errors.poll():
            try:
                message = self.errors.recv()
            except EOFError:
                message = "Reconstruction process exited unexpectedly"
            raise RuntimeError(message)
        if not self.process.is_alive():
            raise RuntimeError(f"Reconstruction process exited ({self.process.exitcode})")
        return self.ring.poll()

    def close(self):
        if not self.closed:
            # A killed worker may leave an Event's internal condition locked forever.
            self.stopped.value = True
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.errors.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
