import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import traceback

import numpy as np

from ito import clock, diagnostics
from ito.protocol import Intrinsics
from ito.render.gpu import MAX_SLOT_COUNT
from ito.render.pose import validate_pose

from .admission import GPU_STAGES, Admission
from .rgbd import RGBDBackend
from .ring import MotionRing, SplatUpdate, UpdateRing
from .slam_vision import RESTART_CAUSES

__all__ = ["Reconstruction", "SplatUpdate", "default_budget"]

log = logging.getLogger(__name__)
STOP_WAIT = 0.5
# Max splats also bounds sorting and drawing per frame: 2^19 kept 90 Hz stereo on an
# RTX 4070 Ti with a full scene. VRAM may hold fewer; the worker budgets that live.
DEFAULT_MAX_SPLATS = 1_000_000


def default_budget(renderer: str = "") -> int:
    if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
        return 16_384
    return DEFAULT_MAX_SPLATS


def _run(recon):
    vram = None
    try:
        robot_first = recon.backend == "slam" and recon.tracking_mode == "robot"
        if robot_first:
            try:
                from .slam import SLAMBackend
            except ImportError as exc:
                raise RuntimeError("Robot-first tracking needs Ito's SLAM components") from exc

            backend = SLAMBackend(
                recon.budget.value,
                recon.intrinsics,
                report=recon.report,
                precision=recon.precision,
                input_resolution=recon.input_resolution,
                precision_report=recon.report_precision,
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
                precision=recon.precision,
                input_resolution=recon.input_resolution,
                precision_report=recon.report_precision,
                **recon.options,
            )
        else:
            backend = RGBDBackend(recon.budget.value, recon.intrinsics, **recon.options)
        backend.bind_admission(recon.admission)
        if backend.xp is not np:
            from .memory import VRAM

            vram = VRAM(backend.xp, sys.modules.get("torch"), recon.notes.send)
        frontend = getattr(backend, "frontend", backend)
        model = getattr(frontend, "model", None)
        network = getattr(model, "network_precision", None)
        next_network_stats = 0
        sequence = 0
        # ITO_TIMELINE names a file for one line per frame the worker took: what became of it.
        timeline = os.environ.get("ITO_TIMELINE")
        timeline = open(timeline, "a", buffering=1) if timeline else None
        cursor = 0
        motion_cursor = 0
        revision = 0
        published_slots = len(backend.keys)
        captured = recon.epoch
        published_capture = recon.epoch
        parent = mp.parent_process()
        while not recon.stopped.value and parent.is_alive():
            now = clock.now() - recon.epoch
            if network is not None and now >= next_network_stats:
                next_network_stats = now + 1
                count = network.statistics()
                if count > recon.nonfinite_points.value:
                    recon.notes.send(dict(event="network_nonfinite", count=count))
                    recon.nonfinite_points.value = count
            backend.expire(now, recon.ring.last_acknowledged())
            limit = recon.budget.value
            if vram is not None:
                fit = vram.check(len(backend.keys), len(backend.keys) <= backend.budget)
                limit = limit if fit is None else min(limit, fit)
                recon.admission.memory_pressure.value = vram.pressure
                recon.admission.gpu_busy.value = -1 if vram.gpu.busy is None else vram.gpu.busy
                recon.vram_free.value = vram.free_bytes
                recon.vram_cached.value = vram.cached_bytes
                recon.cache_releases.value = vram.releases
            if limit != backend.budget:
                retired = backend.resize(limit, now)
                recon.splat_limit.value = limit
                if retired:
                    recon.notes.send(
                        f"Splat limit {limit:,}: the {retired:,} least recently seen fade out"
                    )
            if len(backend.keys) > backend.budget and backend.compact():
                recon.notes.send(f"Splat slots shrunk to {len(backend.keys):,}")
            recon.slot_capacity.value = len(backend.keys)
            if robot_first:
                rows, motion_cursor = recon.motion_ring.read(motion_cursor)
                backend.add_motion(rows)
                while True:
                    try:
                        backend.add_sensors(*recon.sensor_queue.get_nowait())
                    except queue.Empty:
                        break
            frame = None
            take = recon.sequence.value != sequence and recon.admission.available()
            if take and recon.input_lock.acquire(False):
                try:
                    if recon.sequence.value != sequence:
                        previous_sequence = sequence
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
                began = clock.now()
                recon.input_age_seconds.value = max(0, began - captured)
                recon.skipped_inputs.value += max(0, sequence - previous_sequence - 1)
                if timeline:
                    before = (
                        getattr(backend, "tracked", 0),
                        sum(getattr(backend, "restarts", {}).values()),
                        backend.count,
                    )
                    began = clock.now()
                backend.source_sequence = sequence
                backend.source_captured = captured
                dropped = False
                try:
                    with recon.admission.integration():
                        if recon.backend == "slam":
                            if robot_first:
                                backend.integrate(*frame, now, captured=captured)
                            else:
                                backend.integrate(*frame, now, measured=measured)
                        else:
                            backend.integrate(*frame, now)
                except RuntimeError as exc:
                    torch = sys.modules.get("torch")
                    if torch is None or not isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    recon.report_oom(exc)
                    dropped = True
                if dropped:
                    if robot_first:
                        node = backend.graph.current
                        if node is not None and node not in backend.graph.keyframes:
                            frontend.forget([node])
                    tracker = getattr(frontend, "tracker", None)
                    if tracker is not None:
                        tracker.reset_idx_f2k()
                    torch.cuda.empty_cache()
                    if timeline:
                        timeline.write(
                            json.dumps(dict(seq=sequence, outcome="out_of_memory")) + "\n"
                        )
                    continue
                recon.work_seconds.value = clock.now() - began
                recon.work_total.value += recon.work_seconds.value
                recon.integrated.value += 1
                recon.matched.value = getattr(backend, "matched", getattr(backend, "tracked", 0))
                recon.blurred.value = getattr(backend, "blurred", 0)
                recon.scene_slots.value = backend.count
                recon.retiring_slots.value = int(backend.retiring.sum())
                recon.dirty_slots.value = int(backend.dirty.sum())
                if timeline:
                    done = clock.now()
                    restarted = sum(getattr(backend, "restarts", {}).values()) - before[1]
                    tracked = getattr(backend, "tracked", 0) - before[0]
                    since = getattr(backend, "unmatched_since", None)
                    row = dict(
                        event="backend_frame",
                        worker_epoch=recon.epoch,
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
                        splats=int(backend.count),
                        splats_delta=int(backend.count) - before[2],
                        recalls=getattr(backend, "recalls", None),
                        matched=getattr(backend, "matched", None),
                        backend_calls=recon.integrated.value,
                        dirty_slots=recon.dirty_slots.value,
                        retiring_slots=recon.retiring_slots.value,
                        slot_capacity=len(backend.keys),
                        splat_limit=backend.budget,
                        cuda_scope_seconds=recon.admission.active_seconds.value,
                        cuda_stages={
                            name: dict(
                                seconds=recon.admission.stage_seconds[i],
                                calls=recon.admission.stage_calls[i],
                            )
                            for i, name in enumerate(GPU_STAGES)
                        },
                        cuda_memory_peak_bytes=recon.admission.memory_peak.value,
                        gpu_busy_pct=recon.admission.gpu_busy.value,
                        memory_pressure=bool(recon.admission.memory_pressure.value),
                        render_backoffs=recon.admission.overruns.value,
                        input_frames_skipped=max(0, sequence - previous_sequence - 1),
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
            for _ in range(recon.ring.slots):
                dirty = np.flatnonzero(backend.dirty)
                start = np.searchsorted(dirty, cursor)
                ordered = np.concatenate((dirty[start:], dirty[:start]))
                eviction = backend.sequences[ordered] == 0
                indices = np.concatenate((ordered[eviction], ordered[~eviction]))[
                    : recon.ring.batch
                ]
                if not len(indices) and len(backend.keys) == published_slots:
                    break
                contributing = backend.sequences[indices] > 0
                packet_capture = max(
                    published_capture,
                    float(backend.captures[indices[contributing]].max())
                    if contributing.any()
                    else published_capture,
                )
                if not recon.ring.publish(
                    indices,
                    backend.records[indices],
                    len(backend.keys),
                    backend.count,
                    revision,
                    packet_capture,
                    backend.sequences[indices],
                    backend.captures[indices],
                ):
                    break
                published_slots = len(backend.keys)
                published_capture = packet_capture
                retiring = indices[backend.retiring[indices]]
                backend.retire_revision[retiring] = revision
                if len(indices):
                    cursor = (int(indices[-1]) + 1) % len(backend.keys)
                backend.dirty[indices] = False
                revision += 1
            recon.scene_slots.value = backend.count
            recon.retiring_slots.value = int(backend.retiring.sum())
            recon.dirty_slots.value = int(backend.dirty.sum())
            pending_captures = backend.captures[backend.dirty & (backend.sequences > 0)]
            recon.oldest_dirty_capture.value = (
                float(pending_captures.min()) if len(pending_captures) else 0
            )
            if recon.sequence.value == sequence or not recon.admission.available():
                recon.admission.wake.wait(0.005)
    except InterruptedError:
        if not recon.stopped.value:
            recon.errors.send("GPU admission stopped")
    except Exception as exc:
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            recon.report_oom(exc)
            torch.cuda.empty_cache()
        message = str(exc) if recon.backend == "slam" else f"Reconstruction failed: {exc}"
        cause = "".join(traceback.format_exception(exc))
        recon.errors.send(message.splitlines()[0][:500] + "\n" + cause[-6000:])
    finally:
        if vram is not None:
            vram.close()
        recon.errors.close()
        recon.notes.close()


class Reconstruction:
    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        max_splats=1_000_000,
        voxel_size=0.04,
        fade_seconds=0.5,
        device="auto",
        backend="rgbd",
        tracking="robot",
        motion_noise=None,
        frame_delay=0.0,
        exposure=None,
        precision="full",
        input_resolution=100,
    ):
        if not 1 <= max_splats <= MAX_SLOT_COUNT:
            raise ValueError(f"Max splats must be between 1 and {MAX_SLOT_COUNT:,}")
        if not all(np.isfinite(v) and v > 0 for v in (voxel_size, fade_seconds)):
            raise ValueError("Voxel size and fade must be finite and positive")
        if backend not in ("rgbd", "slam"):
            raise ValueError("Reconstruction backend must be rgbd or slam")
        if tracking not in ("robot", "vision"):
            raise ValueError("Tracking must be robot or vision")
        from ito.desktop.settings import AppSettings

        AppSettings(precision=precision, input_resolution=input_resolution)
        self.precision, self.input_resolution = precision, input_resolution
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
        self.active_precision = context.RawValue("b", 0)
        self.oom_count = context.RawValue("Q", 0)
        self.last_oom = context.RawValue("d", 0)
        self.nonfinite_points = context.RawValue("Q", 0)
        self.status_lock = context.Lock()
        self.status_text = context.RawArray("B", 1024)
        self.output_lock = context.Lock()
        self.output_camera = context.RawArray("f", 16)
        self.output_correction = context.RawArray("f", np.eye(4).ravel().tolist())
        self.motion_ring = MotionRing(context)
        self.sensor_queue = context.Queue(maxsize=256)
        self.output_captured = context.RawValue("d", 0)
        self.tracked = context.RawValue("Q", 0)
        self.integrated = context.RawValue("Q", 0)
        self.matched = context.RawValue("Q", 0)
        self.blurred = context.RawValue("Q", 0)
        self.scene_slots = context.RawValue("I", 0)
        self.retiring_slots = context.RawValue("I", 0)
        self.dirty_slots = context.RawValue("I", 0)
        self.oldest_dirty_capture = context.RawValue("d", 0)
        self.work_seconds = context.RawValue("d", 0)
        self.work_total = context.RawValue("d", 0)
        self.vram_free = context.RawValue("Q", 0)
        self.vram_cached = context.RawValue("Q", 0)
        self.cache_releases = context.RawValue("Q", 0)
        self.input_age_seconds = context.RawValue("d", 0)
        self.skipped_inputs = context.RawValue("Q", 0)
        self.tracking = context.RawValue("b", False)
        self.restart_counts = context.RawArray("Q", len(RESTART_CAUSES))
        self.report(
            "Posed RGB-D"
            if backend != "slam"
            else "Starting robot-first tracking"
            if self.tracking_mode == "robot"
            else "Starting MASt3R-SLAM"
        )
        self.ring = UpdateRing(context, self.epoch, fade_seconds, slots=16)
        pixels = intrinsics.width * intrinsics.height
        self.rgb = context.RawArray("B", pixels * 3)
        self.depth = context.RawArray("f", pixels)
        self.camera = context.RawArray("f", 16)
        self.captured = context.RawValue("d", 0)
        self.measured = context.RawValue("b", False)
        self.sequence = context.RawValue("Q", 0)
        self.budget = context.RawValue("I", max_splats)
        self.splat_limit = context.RawValue("I", max_splats)
        self.slot_capacity = context.RawValue("I", 0)
        self.input_lock = context.Lock()
        self.stopped = context.RawValue("b", False)
        self.admission = Admission(context, self.stopped)
        reader, self.errors = context.Pipe(duplex=False)
        notes, self.notes = context.Pipe(duplex=False)
        self.process = None
        process = context.Process(target=_run, args=(self,), name="ito-reconstruction")
        process.start()
        self.process = process
        self.errors.close()
        self.errors = reader
        self.notes.close()
        self.notes = notes
        self.closed = False

    def report_precision(self, active):
        self.active_precision.value = {"fp32": 1, "bf16": 2, "fp16": 3}[active]
        self.notes.send(f"MASt3R precision {active}")

    def report_oom(self, error):
        self.oom_count.value += 1
        self.last_oom.value = clock.now()
        self.notes.send(
            dict(event="network_out_of_memory", count=self.oom_count.value, reason=str(error))
        )

    def statistics(self):
        tracked = self.tracked.value
        from .slam_image import input_size

        return dict(
            slam_precision={0: None, 1: "Full (fp32)", 2: "Half (bf16)", 3: "Half (fp16)"}[
                self.active_precision.value
            ],
            slam_input_size=input_size(
                self.intrinsics.width, self.intrinsics.height, self.input_resolution
            ),
            reconstruction_oom=self.last_oom.value > 0 and clock.now() - self.last_oom.value < 10,
            network_oom_count=self.oom_count.value,
            network_nonfinite_points=self.nonfinite_points.value,
            worker_epoch=self.epoch,
            backend_calls=self.integrated.value,
            matched_pct=100 * self.matched.value / tracked
            if tracked and self.tracking_mode == "robot"
            else None,
            blurred_frames=self.blurred.value,
            slot_count=self.scene_slots.value,
            slot_capacity=self.slot_capacity.value,
            splat_limit=self.splat_limit.value,
            max_splats=self.budget.value,
            slot_fill_pct=100 * self.scene_slots.value / max(1, self.splat_limit.value),
            retiring_slots=self.retiring_slots.value,
            dirty_slots=self.dirty_slots.value,
            oldest_dirty_age_ms=(clock.now() - self.oldest_dirty_capture.value) * 1000
            if self.oldest_dirty_capture.value
            else None,
            backend_work_ms=self.work_seconds.value * 1000,
            backend_work_total_s=self.work_total.value,
            vram_free_mib=self.vram_free.value / 2**20,
            vram_cached_mib=self.vram_cached.value / 2**20,
            cache_releases=self.cache_releases.value,
            input_age_ms=self.input_age_seconds.value * 1000,
            skipped_input_frames=self.skipped_inputs.value,
            cuda_scope_seconds=self.admission.active_seconds.value,
            cuda_stages={
                name: dict(
                    seconds=self.admission.stage_seconds[i], calls=self.admission.stage_calls[i]
                )
                for i, name in enumerate(GPU_STAGES)
            },
            cuda_completed_chunks=self.admission.chunks.value,
            cuda_memory_peak_bytes=self.admission.memory_peak.value,
            memory_pressure=bool(self.admission.memory_pressure.value),
            render_backoffs=self.admission.overruns.value,
            throttle_remaining_ms=max(0, self.admission.pause_until.value - clock.now()) * 1000,
        )

    def display_completed(self, began, period):
        self.admission.completed(began, period)

    @property
    def max_splats(self):
        return self.budget.value

    def set_max_splats(self, value):
        if not 1 <= value <= MAX_SLOT_COUNT:
            raise ValueError(f"Max splats must be between 1 and {MAX_SLOT_COUNT:,}")
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
        if not self.output_lock.acquire(False):
            return None
        try:
            return np.frombuffer(self.output_correction, np.float32).reshape(4, 4).copy()
        finally:
            self.output_lock.release()

    def motion(self, samples):
        if self.tracking_mode == "robot" and self.backend == "slam":
            self.motion_ring.put(samples)

    def sensors(self, imu, ranges):
        if self.closed or self.tracking_mode != "robot" or self.backend != "slam":
            return
        for _ in range(2):
            try:
                self.sensor_queue.put_nowait((imu, ranges))
                return
            except queue.Full:
                try:
                    self.sensor_queue.get_nowait()
                except queue.Empty:
                    pass

    def restarts(self):
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
            self.admission.wake.notify()
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
        try:
            while self.notes.poll():
                note = self.notes.recv()
                if isinstance(note, dict):
                    fields = {k: v for k, v in note.items() if k != "event"}
                    diagnostics.event(note["event"], **fields)
                    log.warning("Reconstruction: %s", note)
                else:
                    log.info("Reconstruction: %s", note)
        except (EOFError, OSError):
            pass
        if not self.process.is_alive():
            raise RuntimeError(f"Reconstruction process exited ({self.process.exitcode})")
        return self.ring.poll()

    def close(self):
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
            self.sensor_queue.cancel_join_thread()
            self.sensor_queue.close()
            self.errors.close()
            self.notes.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
