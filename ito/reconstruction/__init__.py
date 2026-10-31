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
from .ring import SplatUpdate, UpdateRing

__all__ = ["CLOSE_WAIT", "Reconstruction", "SplatUpdate", "default_budget"]

log = logging.getLogger(__name__)
# Stopping a worker: it gets STOP_WAIT to leave its loop (a step that never ends is killed),
# then EXIT_WAIT to clean up and exit: SLAM's backend thread (at most slam.BACKEND_WAIT) and
# CUDA's teardown (0.35-0.55 s with 9 GB in use, RTX 4070 Ti); a kill gets KILL_WAIT. CLOSE_WAIT
# is all of it: what close() may take, and so what its callers wait for.
STOP_WAIT = 0.5
EXIT_WAIT = 3.0
KILL_WAIT = 1.0
CLOSE_WAIT = STOP_WAIT + EXIT_WAIT + KILL_WAIT
# Max splats also bounds sorting and drawing per frame: 2^19 kept 90 Hz stereo on an
# RTX 4070 Ti with a full scene. VRAM may hold fewer; the worker budgets that live.
DEFAULT_MAX_SPLATS = 1_000_000


def default_budget(renderer: str = "") -> int:
    if any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
        return 16_384
    return DEFAULT_MAX_SPLATS


def _run(recon):
    vram = backend = timeline = None
    try:
        if recon.backend == "slam":
            from .slam import SLAMBackend

            backend = SLAMBackend(
                recon.budget.value,
                recon.intrinsics,
                report=recon.report,
                precision=recon.precision,
                input_resolution=recon.input_resolution,
                keyframes=recon.keyframes,
                precision_report=recon.report_precision,
                **recon.options,
            )
        else:
            backend = RGBDBackend(recon.budget.value, recon.intrinsics, **recon.options)
        backend.bind_admission(recon.admission)
        if backend.xp is not np:
            from .memory import VRAM

            vram = VRAM(backend.xp, sys.modules.get("torch"), recon.notes.send)
        model = getattr(backend, "model", None)
        network = getattr(model, "network_precision", None)
        next_network_stats = 0
        sequence = 0
        # ITO_TIMELINE names a file for one line per frame the worker took: what became of it.
        path = os.environ.get("ITO_TIMELINE")
        timeline = open(path, "a", buffering=1) if path else None
        if timeline:
            import psutil

            worker = psutil.Process()
        cursor = 0
        revision = 0
        published_slots = len(backend.keys)
        captured = recon.epoch
        published_capture = recon.epoch
        parent = mp.parent_process()
        posed = [0.0]

        def publish_pose():
            """The tracked camera pose to the pilot. SLAM calls it as soon as the frame is
            tracked, before redrawing its splat map (~10 ms), so the view need not wait for it."""
            if recon.output_lock.acquire(False):
                try:
                    np.frombuffer(recon.output_camera, np.float32)[:] = backend.camera_pose.ravel()
                    recon.output_captured.value = captured
                    recon.tracked.value = backend.tracked
                    recon.tracking.value = not backend.lost and backend.tracked > 0
                    recon.display_scale.value = backend.tof.value or 0.0
                finally:
                    recon.output_lock.release()
                posed[0] = posed[0] or clock.now()

        if recon.backend == "slam":
            backend.pose_ready = publish_pose
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
                if vram.pressure and hasattr(backend, "vram_full"):
                    backend.vram_full(now, vram.free_bytes)
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
            while recon.backend == "slam":
                try:
                    backend.add_ranges(recon.range_queue.get_nowait())
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
                finally:
                    recon.input_lock.release()
            if frame is not None:
                began = clock.now()
                recon.input_age_seconds.value = max(0, began - captured)
                recon.skipped_inputs.value += max(0, sequence - previous_sequence - 1)
                if timeline:
                    before = (getattr(backend, "tracked", 0), backend.count)
                    began = clock.now()
                posed[0] = 0.0
                backend.source_sequence = sequence
                backend.source_captured = captured
                dropped = False
                try:
                    with recon.admission.integration():
                        backend.integrate(*frame, now)
                except RuntimeError as exc:
                    torch = sys.modules.get("torch")
                    if torch is None or not isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
                    recon.report_oom(exc)
                    dropped = True
                if dropped:
                    tracker = getattr(backend, "tracker", None)
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
                recon.scene_slots.value = backend.count
                recon.retiring_slots.value = int(backend.retiring.sum())
                recon.dirty_slots.value = int(backend.dirty.sum())
                if timeline:
                    done = clock.now()
                    tracked = getattr(backend, "tracked", 0) - before[0]
                    row = dict(
                        event="backend_frame",
                        worker_epoch=recon.epoch,
                        seq=sequence,
                        captured=round(captured, 4),
                        began=round(began, 4),
                        age_ms=round((began - captured) * 1000, 1),
                        work_ms=round((done - began) * 1000, 1),
                        outcome="integrated"
                        if recon.backend != "slam"
                        else "tracked"
                        if tracked
                        else "lost",
                        splats=int(backend.count),
                        splats_delta=int(backend.count) - before[1],
                        keyframes=len(backend.keyframes) if recon.backend == "slam" else None,
                        # Of work_ms: the splat map's redraws of SLAM's keyframes.
                        draw_ms=round(getattr(backend, "draw_seconds", 0) * 1000, 1),
                        # From began until the pilot could read the frame's camera pose.
                        pose_ms=round((posed[0] - began) * 1000, 1) if posed[0] else None,
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
                        # CuPy's pool and Torch's reserved blocks; the process's
                        # working set. Flat over a session once SLAM's buffers are allocated.
                        cuda_memory_bytes=backend.cuda_memory_bytes()
                        if backend.xp is not np
                        else None,
                        torch_allocated_bytes=sys.modules["torch"].cuda.memory_allocated()
                        if "torch" in sys.modules and backend.xp is not np
                        else None,
                        working_set_bytes=(memory := worker.memory_info()).rss,
                        # Windows: commit, which also counts what the driver backs with RAM.
                        private_bytes=getattr(memory, "private", None),
                        gpu_busy_pct=recon.admission.gpu_busy.value,
                        memory_pressure=bool(recon.admission.memory_pressure.value),
                        render_backoffs=recon.admission.overruns.value,
                        input_frames_skipped=max(0, sequence - previous_sequence - 1),
                    )
                    timeline.write(json.dumps(row) + "\n")
                if recon.backend == "slam":
                    # Again after the splat map's work: the early write may have found the pilot
                    # holding the lock.
                    publish_pose()
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
        recon.leaving.value = True  # out of the loop: close() now waits EXIT_WAIT, not STOP_WAIT
        # The backend's own threads (SLAM's global solve): once stopped, their GPU work ends at
        # its next admission chunk.
        if backend is not None and hasattr(backend, "close"):
            backend.close()
        if timeline:
            timeline.close()
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
        precision="full",
        input_resolution=100,
        keyframes=128,
    ):
        if not 1 <= max_splats <= MAX_SLOT_COUNT:
            raise ValueError(f"Max splats must be between 1 and {MAX_SLOT_COUNT:,}")
        if not all(np.isfinite(v) and v > 0 for v in (voxel_size, fade_seconds)):
            raise ValueError("Voxel size and fade must be finite and positive")
        if backend not in ("rgbd", "slam"):
            raise ValueError("Reconstruction backend must be rgbd or slam")
        from ito.desktop.settings import AppSettings

        AppSettings(precision=precision, input_resolution=input_resolution, keyframes=keyframes)
        self.precision, self.input_resolution = precision, input_resolution
        self.keyframes = keyframes
        self.backend = backend
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
        self.output_captured = context.RawValue("d", 0)
        # ToF ranges for the map's display scale, until the worker has fixed it (then nonzero).
        self.range_queue = context.Queue(maxsize=64)
        self.display_scale = context.RawValue("d", 0)
        self.tracked = context.RawValue("Q", 0)
        self.integrated = context.RawValue("Q", 0)
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
        self.report("Posed RGB-D" if backend != "slam" else "Starting MASt3R-SLAM")
        self.ring = UpdateRing(context, self.epoch, fade_seconds, slots=16)
        pixels = intrinsics.width * intrinsics.height
        self.rgb = context.RawArray("B", pixels * 3)
        self.depth = context.RawArray("f", pixels)
        self.camera = context.RawArray("f", 16)
        self.captured = context.RawValue("d", 0)
        self.sequence = context.RawValue("Q", 0)
        self.budget = context.RawValue("I", max_splats)
        self.splat_limit = context.RawValue("I", max_splats)
        self.slot_capacity = context.RawValue("I", 0)
        self.input_lock = context.Lock()
        self.stopped = context.RawValue("b", False)
        self.leaving = context.RawValue("b", False)  # the worker left its loop, cleans up
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
        from .slam import input_shape

        return dict(
            slam_precision={0: None, 1: "Full (fp32)", 2: "Half (bf16)", 3: "Half (fp16)"}[
                self.active_precision.value
            ],
            slam_input_size=input_shape(
                self.intrinsics.width, self.intrinsics.height, self.input_resolution
            ),
            slam_keyframes=self.keyframes,
            slam_display_scale=self.display_scale.value or None,
            reconstruction_oom=self.last_oom.value > 0 and clock.now() - self.last_oom.value < 10,
            network_oom_count=self.oom_count.value,
            network_nonfinite_points=self.nonfinite_points.value,
            worker_epoch=self.epoch,
            backend_calls=self.integrated.value,
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

    def ranges(self, grids):
        """ToF range grids (see tof.RangeScale.add), on the pilot clock. A full queue drops the
        oldest; once the map's scale is fixed, nothing is sent."""
        if self.closed or self.backend != "slam" or self.display_scale.value:
            return
        for grid in grids:
            for _ in range(2):
                try:
                    self.range_queue.put_nowait(grid)
                    break
                except queue.Full:
                    try:
                        self.range_queue.get_nowait()
                    except queue.Empty:
                        pass

    def submit(self, rgb, depth=None, camera=None, captured_at=None):
        """Depth is axial metres; pose is world-from-camera (+Y up, -Z forward).

        SLAM needs no depth, and takes the pose only as the robot's own estimate, and only
        the first frame's: where to put the map (gravity, start heading and position).
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
            deadline = clock.now() + STOP_WAIT
            while self.process.is_alive() and not self.leaving.value and clock.now() < deadline:
                self.process.join(timeout=0.01)
            if self.leaving.value:
                self.process.join(timeout=EXIT_WAIT)
            if self.process.is_alive():
                # SIGTERM stays pending on a stopped process; SIGKILL does not.
                log.warning(
                    "Reconstruction process %d killed: %s",
                    self.process.pid,
                    f"still exiting {EXIT_WAIT:.1f} s after it stopped"
                    if self.leaving.value
                    else f"still busy {STOP_WAIT:.1f} s after stop",
                )
                self.process.kill()
                self.process.join(timeout=KILL_WAIT)
                if self.process.is_alive():
                    log.error("Reconstruction process %d did not exit", self.process.pid)
            self.range_queue.cancel_join_thread()
            self.range_queue.close()
            self.errors.close()
            self.notes.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
