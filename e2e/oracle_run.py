"""Real-time oracle intervention with concurrent production GL stereo and CUDA mapping.

DISPLAY=:0 GALLIUM_DRIVER=d3d12 PYTHONPATH=. flock /mnt/c/Users/me/Projects/ito/ceres.lock \
  .venv/bin/python e2e/oracle_run.py RECORDING --arm TT --out OUT --record
TT/TE/ET/EE select true/estimated pose then depth; TF uses true pose and sparse ToF.
main-TT/main-EE use unchanged main SLAMBackend.place with those exact frozen inputs.
Recording is a separate run from timing; no image readback in timing runs.
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ito.protocol import Intrinsics
from ito.reconstruction.admission import Admission
from ito.reconstruction.ring import UpdateRing


class Publication:
    def __init__(self, budget):
        self.budget = budget
        self.records = np.zeros((0, 4, 4), np.float32)
        self.owners = np.zeros(0, np.int64)
        self.dirty = np.zeros(0, bool)
        self.nodes = {}
        self.cursor = self.revision = 0

    def update(self, maps):
        old = {node: self.nodes.pop(node, np.empty(0, np.int64)) for node in maps}
        for slots in old.values():
            self.owners[slots] = -1
            self.records[slots] = 0
            self.dirty[slots] = True
        reserved = {node: old[node][: len(records)] for node, records in maps.items()}
        for node, slots in reserved.items():
            self.owners[slots] = node
        needed = sum(max(0, len(records) - len(reserved[node])) for node, records in maps.items())
        missing = max(0, needed - np.count_nonzero(self.owners < 0))
        if missing:
            capacity = max(
                len(self.owners) + missing, min(self.budget, max(16384, 2 * len(self.owners)))
            )
            n = capacity - len(self.owners)
            self.records = np.pad(self.records, ((0, n), (0, 0), (0, 0)))
            self.owners = np.pad(self.owners, (0, n), constant_values=-1)
            self.dirty = np.pad(self.dirty, (0, n))
        free = np.flatnonzero(self.owners < 0)
        cursor = 0
        for node, records in maps.items():
            n = max(0, len(records) - len(reserved[node]))
            slots = np.concatenate((reserved[node], free[cursor : cursor + n]))
            cursor += n
            self.nodes[node] = slots
            self.owners[slots] = node
            self.records[slots] = records
            self.dirty[slots] = True

    def flush(self, ring, captured):
        dirty = np.flatnonzero(self.dirty)
        split = np.searchsorted(dirty, self.cursor)
        for indices in (dirty[split:], dirty[:split]):
            for start in range(0, len(indices), ring.batch):
                ids = indices[start : start + ring.batch]
                if not ring.publish(
                    ids,
                    self.records[ids],
                    len(self.records),
                    np.count_nonzero(self.owners >= 0),
                    self.revision,
                    captured,
                ):
                    return
                self.dirty[ids] = False
                self.revision += 1
                self.cursor = int(ids[-1]) + 1


class MainPublication:
    """The same stable-slot eviction-first publication loop as Reconstruction._run."""

    def __init__(self, backend):
        self.backend = backend
        self.cursor = self.revision = 0
        self.slots = -1

    def flush(self, ring, captured):
        backend = self.backend
        for _ in range(ring.slots):
            dirty = np.flatnonzero(backend.dirty)
            split = np.searchsorted(dirty, self.cursor)
            ordered = np.concatenate((dirty[split:], dirty[:split]))
            eviction = backend.sequences[ordered] == 0
            indices = np.concatenate((ordered[eviction], ordered[~eviction]))[: ring.batch]
            if not len(indices) and self.slots == len(backend.keys):
                break
            if not ring.publish(
                indices,
                backend.records[indices],
                len(backend.keys),
                backend.count,
                self.revision,
                captured,
                backend.sequences[indices],
                backend.captures[indices],
            ):
                break
            self.slots = len(backend.keys)
            backend.retire_revision[indices[backend.retiring[indices]]] = self.revision
            backend.dirty[indices] = False
            if len(indices):
                self.cursor = int(indices[-1]) + 1
            self.revision += 1


def worker(args, ring, admission, ready, start, stopped, status):
    try:
        import cupy as cp
        import torch
        from oracle_sensor import tof

        from ito.reconstruction.mapper import GaussianMap, Keyframe
        from ito.reconstruction.memory import VRAM
        from ito.reconstruction.rgbd import RGBDBackend
        from ito.reconstruction.slam import SLAMBackend
        from ito.reconstruction.slam_image import SLAMImage

        torch.set_num_threads(2)
        torch.manual_seed(40)
        path = args.recording / "source"
        rows = json.loads((args.estimates / "frames.json").read_text())
        source_rows = [json.loads(s) for s in (path / "frames.jsonl").read_text().splitlines()]
        settings = json.loads((args.estimates / "settings.json").read_text())
        if args.precision:
            settings["precision"] = args.precision
        k = Intrinsics.model_validate_json((path / "camera.json").read_text())
        camera_image = SLAMImage(k, settings["input_resolution"])
        first = np.load(args.estimates / rows[0]["file"])
        alignment = first["pose"] @ np.linalg.inv(first["true_pose"])
        origin = settings["origin"]
        # All arms, including truth, see the same frozen estimates' availability schedule.
        zero = rows[0]["available"]
        for row in rows:
            row["release"] = row["available"] - zero
        main = args.arm.startswith("main-")
        arm = args.arm.removeprefix("main-")
        if main:
            engine = RGBDBackend(args.max_splats, k, device="cuda")
            engine.torch = torch
            if getattr(args, "depth_support", False):
                from ito.reconstruction.depth_support import DepthSupport

                engine.depth_support = DepthSupport()
        else:
            engine = GaussianMap(args.max_splats, precision=settings["precision"])
        admission.bind_stream(cp)
        admission.memory_used = lambda: (
            torch.cuda.memory_reserved() + cp.get_default_memory_pool().total_bytes()
        )
        vram = VRAM(cp, torch, print, slot_bytes=96 if main else 400)
        pub = MainPublication(engine) if main else Publication(args.max_splats)
        rng = np.random.default_rng(40)
        step_costs, add_costs = [], []
        log = (args.out / "mapping.jsonl").open("w", buffering=1)
        admitted = []
        cursor = 0
        last_publish = -1
        captured = 0
        oom = 0
        # JIT/kernel warm-up is startup, never charged as observation time.
        if not main:
            rgb = first["rgb"]
            z = first["true_depth"]
            y, x = np.mgrid[: k.height, : k.width]
            points = np.stack(((x - k.cx) / k.fx * z, -(y - k.cy) / k.fy * z, -z), -1)
            frame = Keyframe(
                0, rgb, alignment @ first["true_pose"], k, points, rgb, np.ones(z.shape), 0, depth=z
            )
            with admission.integration():
                engine.add(frame)
                engine.step()
            engine = GaussianMap(args.max_splats, precision=settings["precision"])
        if not main and args.oom_at == -2:
            refine = engine._refine_parameters
            failed = False

            def fail_refine():
                nonlocal failed
                refine()
                if not failed:
                    failed = True
                    torch.empty(1 << 60, device="cuda", dtype=torch.uint8)

            engine._refine_parameters = fail_refine
        cp.cuda.get_current_stream().synchronize()
        ready.set()
        while not start.value and not stopped.value:
            time.sleep(0.002)
        epoch = start.value
        while not stopped.value:
            now = time.perf_counter() - epoch
            if now > args.seconds:
                break
            if main:
                engine.expire(now, ring.last_acknowledged())
                if len(engine.keys) > engine.budget:
                    engine.compact()
            try:
                with admission.integration():
                    fit = vram.check(
                        len(engine.keys) if main else engine.count,
                        not main or len(engine.keys) <= engine.budget,
                    )
                    limit = args.max_splats if fit is None else min(args.max_splats, fit)
                    if limit != engine.budget:
                        engine.resize(limit, now) if main else engine.resize(limit)
                    if vram.pressure and not main and len(engine.views) > 2:
                        del engine.views[next(iter(engine.views))]
                    # Fixed observations are retained, not substituted with an arm's faster frames.
                    if cursor < len(rows) and rows[cursor]["release"] <= now:
                        row = rows[cursor]
                        data = np.load(args.estimates / row["file"])
                        pose = alignment @ data["true_pose"] if arm[0] == "T" else data["pose"]
                        depth = data["true_depth"] if arm[1] == "T" else data["depth"]
                        tof_source = None
                        if arm == "TF":
                            rawrow = dict(row, captured=row["captured"], file=row["file"])
                            depth, tof_source = tof(source_rows, path, rawrow, k, origin, rng)
                        rgb = data["rgb"]
                        began = time.perf_counter()
                        try:
                            if args.oom_at == cursor and main:
                                # WDDM can overcommit more than dedicated VRAM. Request beyond
                                # the addressable device allocation, without changing any cap.
                                torch.empty(1 << 60, device="cuda", dtype=torch.uint8)
                                raise AssertionError("CUDA allocator did not reject the fault")
                            if main:
                                import cv2

                                sw, sh = camera_image.scaled
                                d = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_NEAREST)
                                c = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
                                fx, fy = k.fx * sw / k.width, k.fy * sh / k.height
                                cx, cy = k.cx * sw / k.width, k.cy * sh / k.height
                                y, x = np.mgrid[:sh, :sw]
                                points = np.stack(((x - cx) / fx * d, -(y - cy) / fy * d, -d), -1)
                                cloud = SimpleNamespace(
                                    points=torch.as_tensor(
                                        points.reshape(-1, 3), device="cuda", dtype=torch.float32
                                    ),
                                    colors=torch.as_tensor(c.reshape(-1, 3), device="cuda"),
                                    valid=torch.as_tensor(
                                        (np.isfinite(d) & (d > 0)).ravel(), device="cuda"
                                    ),
                                    intrinsics=(fx, fy, cx, cy),
                                    shape=(sh, sw),
                                )
                                engine.source_sequence = row["source_frame_id"] + 1
                                engine.source_captured = row["captured"]
                                SLAMBackend.place(engine, cloud, pose, now)
                            else:
                                y, x = np.mgrid[: k.height, : k.width]
                                z = depth
                                points = np.stack(
                                    ((x - k.cx) / k.fx * z, -(y - k.cy) / k.fy * z, -z), -1
                                )
                                seed = engine._seed
                                if args.oom_at == cursor:

                                    def fail_after_seed(*values, seed=seed):
                                        seed(*values)
                                        torch.empty(1 << 60, device="cuda", dtype=torch.uint8)
                                        raise AssertionError("CUDA allocator accepted the fault")

                                    engine._seed = fail_after_seed
                                try:
                                    engine.add(
                                        Keyframe(
                                            row["source_frame_id"],
                                            rgb,
                                            pose,
                                            k,
                                            points,
                                            rgb,
                                            (np.isfinite(z) & (z > 0)).astype(np.float32),
                                            row["captured"],
                                            depth=z,
                                        )
                                    )
                                finally:
                                    engine._seed = seed
                            captured = row["captured"]
                            admitted.append(row["source_frame_id"])
                        except torch.cuda.OutOfMemoryError:
                            args.oom_warning.value = time.perf_counter()
                            oom += 1
                            torch.cuda.empty_cache()
                            # Precision never changes: the incoming observation is dropped.
                            if not main:
                                engine.views.pop(row["source_frame_id"], None)
                            print(
                                "Reconstruction is degraded: mapper dropped a frame (CUDA OOM)",
                                flush=True,
                            )
                        add_costs.append((time.perf_counter() - began) * 1000)
                        log.write(
                            json.dumps(
                                dict(
                                    event="observation",
                                    source_frame_id=row["source_frame_id"],
                                    released=row["release"],
                                    started=now,
                                    finished=time.perf_counter() - epoch,
                                    captured=row["captured"],
                                    tof_source=tof_source,
                                    oom=oom,
                                )
                            )
                            + "\n"
                        )
                        cursor += 1
                    if not main:
                        began = time.perf_counter()
                        try:
                            engine.step()
                        except torch.cuda.OutOfMemoryError:
                            args.oom_warning.value = time.perf_counter()
                            oom += 1
                            engine.recover()
                            torch.cuda.empty_cache()
                            print(
                                "Reconstruction is degraded: training frame dropped (CUDA OOM)",
                                flush=True,
                            )
                        step_costs.append((time.perf_counter() - began) * 1000)
                    if now - last_publish >= 0.1 and admitted:
                        if not main:
                            pub.update(engine.records_for(engine.keys()))
                        last_publish = now
                        log.write(
                            json.dumps(
                                dict(
                                    event="publication",
                                    at=time.perf_counter() - epoch,
                                    observations=admitted.copy(),
                                    count=engine.count,
                                    steps=getattr(engine, "steps", 0),
                                    limit=engine.budget,
                                    vram_free=vram.free_bytes,
                                    oom=oom,
                                )
                            )
                            + "\n"
                        )
                    pub.flush(ring, captured)
            except torch.cuda.OutOfMemoryError:
                oom += 1
                args.oom_warning.value = time.perf_counter()
                if not main:
                    engine.recover()
                torch.cuda.empty_cache()
                print("Reconstruction is degraded: GPU update dropped (CUDA OOM)", flush=True)
                time.sleep(0.005)
            if main or not admitted:
                time.sleep(0.001)
        (args.out / "worker.json").write_text(
            json.dumps(
                dict(
                    admitted=admitted,
                    step_ms=step_costs,
                    add_ms=add_costs,
                    oom=oom,
                    torch_peak_reserved=torch.cuda.max_memory_reserved(),
                    cuda_chunks=admission.chunks.value,
                    precision=settings["precision"],
                    loss_dtype=str(getattr(engine, "loss_dtype", None)),
                    alignment=alignment.tolist(),
                    zero=zero,
                    source_origin=origin,
                    estimate_settings=settings,
                ),
                indent=2,
            )
        )
        log.close()
        vram.close()
    except BaseException:
        status.put(traceback.format_exc())
        ready.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--estimates", type=Path)
    parser.add_argument(
        "--arm", choices=("TT", "TE", "ET", "EE", "TF", "main-TT", "main-EE"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument(
        "--view-start",
        type=float,
        help="Source-relative display start, independent of estimate availability (A/B views)",
    )
    parser.add_argument("--max-splats", type=int, default=1_000_000)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--depth-support",
        action="store_true",
        help="Use cross-view depth agreement before point-map carving",
    )
    parser.add_argument("--precision", choices=("full", "half"))
    parser.add_argument(
        "--oom-at",
        type=int,
        default=-1,
        help="real allocator OOM after observation N seeds; -2 after first refinement; -1 disables",
    )
    parser.add_argument("--eye-size", type=int, default=2160)
    args = parser.parse_args()
    args.recording, args.out = args.recording.resolve(), args.out.resolve()
    args.estimates = args.estimates or args.recording / "causal-full"
    args.out.mkdir(parents=True, exist_ok=True)
    import moderngl
    import pygame

    from ito.desktop.record import Recorder
    from ito.render.renderer import GaussianRenderer

    context = mp.get_context("spawn")
    stopped = context.RawValue("b", False)
    start = context.RawValue("d", 0)
    ready = context.Event()
    # Stop between integrations; interruption inside one would skip its final audit report.
    admission = Admission(context, context.RawValue("b", False))
    ring = UpdateRing(context, 1, 0.5, slots=16)
    status = context.Queue()
    args.oom_warning = context.RawValue("d", -1)
    child = context.Process(
        target=worker, args=(args, ring, admission, ready, start, stopped, status)
    )
    child.start()
    pygame.display.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.set_mode((640, 640), pygame.OPENGL | pygame.DOUBLEBUF)
    gl = moderngl.create_context(require=430)
    renderer = GaussianRenderer(gl)
    from ito.desktop.overlay import Overlay, PilotStatus

    overlay = Overlay(gl, args.max_splats)
    warning_saved = False
    warning_frames = 0
    k = Intrinsics.model_validate_json((args.recording / "source/camera.json").read_text())
    from measure_gl import projection

    stereo_k = Intrinsics(
        width=args.eye_size,
        height=args.eye_size,
        fx=args.eye_size / 2,
        fy=args.eye_size / 2,
        cx=args.eye_size / 2,
        cy=args.eye_size / 2,
    )
    target = gl.simple_framebuffer((args.eye_size, args.eye_size), components=4, dtype="f1")
    camera_target = gl.simple_framebuffer((k.width, k.height), components=4, dtype="f1")
    sources = [
        json.loads(s) for s in (args.recording / "source/frames.jsonl").read_text().splitlines()
    ]
    settings = json.loads((args.estimates / "settings.json").read_text())
    settings["depth_support"] = args.depth_support
    if args.precision:
        settings["precision"] = args.precision
    rows = json.loads((args.estimates / "frames.json").read_text())
    zero = rows[0]["available"]
    view_start = zero if args.view_start is None else args.view_start
    settings["view_start"] = view_start
    times = np.array([r["captured"] - settings["origin"] - view_start for r in sources])
    first = np.load(args.estimates / rows[0]["file"])
    alignment = first["pose"] @ np.linalg.inv(first["true_pose"])
    captures = []
    recorder = Recorder(args.out / "motion", fps=10) if args.record else None
    records = np.zeros((0, 4, 4), np.float32)
    costs, service, misses, longest, consecutive = [], [], 0, 0, 0
    next_record = 0
    # Cold prefixes and fixed motion events are selected from source, before mapper results.
    events = [0, 5, 10, 15] if args.seconds <= 25 else [0, 12, 50, 76]
    snapshots = [a + b for a in events for b in (0.1, 0.5, 1, 2) if a + b < args.seconds]
    saved = set()
    try:
        while not ready.wait(0.1):
            pygame.event.pump()
            if not child.is_alive():
                raise RuntimeError("Mapper failed during startup")
        if not status.empty():
            raise RuntimeError(status.get())
        start.value = time.perf_counter()
        ring.epoch = start.value
        deadline = start.value
        previous_index = -1
        last_completion = 0
        prior_completion = 0
        prior_records = records.copy()
        frame = None
        while (now := time.perf_counter() - start.value) < args.seconds:
            pygame.event.pump()
            if not status.empty():
                raise RuntimeError(status.get())
            for moment in snapshots:
                if now >= moment and moment not in saved:
                    if recorder:
                        reference = int(
                            np.clip(
                                np.searchsorted(times, moment, side="right") - 1,
                                0,
                                len(sources) - 1,
                            )
                        )
                        if last_completion <= moment:
                            captures.append((moment, last_completion, reference, records.copy()))
                        else:
                            captures.append((moment, prior_completion, reference, prior_records))
                    saved.add(moment)
            if recorder:
                prior_records = records.copy()
                prior_completion = last_completion
            began = time.perf_counter()
            for _ in range(2):
                packet = ring.poll()
                if packet is None:
                    break
                if len(records) != packet.capacity:
                    r = np.zeros((packet.capacity, 4, 4), np.float32)
                    r[: min(len(r), len(records))] = records[: len(r)]
                    records = r
                retiring = packet.records[:, 3, 3] == 1
                packet.records[retiring, 1, 3] = np.minimum(
                    packet.records[retiring, 1, 3], time.perf_counter() - start.value + 0.5
                )
                if not renderer.apply(packet):
                    raise RuntimeError("GL upload rejected a packet")
                records[packet.indices] = packet.records
            index = int(np.clip(np.searchsorted(times, now, side="right") - 1, 0, len(sources) - 1))
            if index != previous_index:
                frame = np.load(args.recording / "source" / sources[index]["file"])
                previous_index = index
            camera = alignment @ frame["camera"]
            # Moving the current head is independent of camera/reconstruction publication.
            head = np.eye(4, dtype=np.float32)
            head[0, 3] = 0.03 * math.sin(now * 2)
            renderer.sort(camera.astype(np.float32), head)
            for eye in (-0.032, 0.032):
                offset = head.copy()
                offset[0, 3] += eye
                renderer.draw(
                    camera.astype(np.float32),
                    offset,
                    projection(stereo_k),
                    target,
                    sort=False,
                    clear=(0, 0, 0, 0),
                )
            gl.finish()
            ended = time.perf_counter()
            elapsed = ended - deadline
            missed = max(0, math.ceil(elapsed * 90) - 1)
            misses += missed
            consecutive = consecutive + missed if missed else 0
            longest = max(longest, consecutive)
            costs.append(elapsed * 1000)
            service.append((ended - began) * 1000)
            admission.completed(deadline, 1 / 90)
            if recorder and now >= next_record:
                renderer.draw(
                    camera.astype(np.float32),
                    np.eye(4, dtype=np.float32),
                    projection(k),
                    camera_target,
                    sort=True,
                    clear=(0, 0, 0, 0),
                )
                pixels = camera_target.read(components=3)
                recorder.pilot(
                    now,
                    pixels,
                    (k.width, k.height),
                    arm=args.arm,
                    source_frame_id=sources[index]["source_frame_id"],
                    scene_age_ms=None,
                    settings=settings,
                )
                recorder.camera(frame["rgb"], index, now, camera)
                next_record += 0.1
            if 0 <= time.perf_counter() - args.oom_warning.value < 10:
                warning_frames += 1
                gl.screen.use()
                gl.screen.clear(0.025, 0.035, 0.055, 1)
                overlay.begin([], (640, 640), False)
                overlay.draw(
                    PilotStatus(
                        robot="Oracle mapper experiment",
                        reconstruction=f"Mapper {settings['precision']}; allocation recovered",
                    ),
                    90,
                    renderer.count,
                    None,
                    False,
                    None,
                    live=True,
                    performance={
                        "reconstruction_oom": True,
                    },
                    target=gl.screen,
                )
                if not warning_saved and warning_frames >= 2:
                    from OpenGL import GL
                    from PIL import Image

                    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
                    GL.glReadBuffer(GL.GL_BACK)
                    pixels = GL.glReadPixels(0, 0, 640, 640, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
                    image = Image.frombytes("RGB", (640, 640), pixels)
                    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
                        args.out / "oom-warning.png"
                    )
                    warning_saved = True
                pygame.display.flip()
            last_completion = ended - start.value
            deadline += max(1, missed + 1) / 90
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        if not status.empty():
            raise RuntimeError(status.get())
    finally:
        stopped.value = True
        child.join(10)
        if child.is_alive():
            child.kill()
            child.join()
        if recorder:
            recorder.close()
        if overlay:
            overlay.close()
        renderer.close()
        target.release()
        camera_target.release()
        gl.release()
        pygame.quit()
    for moment, at, index, snapshot in captures:
        # CPU snapshots must include the fade the live GL shader actually displayed.
        retiring = snapshot[:, 3, 3] == 1
        snapshot[retiring, 0, 3] *= np.clip((snapshot[retiring, 1, 3] - at) / 0.5, 0, 1)
        snapshot[:, 3, 3] = 0
        snapshot[:, 1, 3] = 1e30
        np.savez_compressed(
            args.out / f"snapshot-{moment:.1f}.npz",
            records=snapshot,
            deadline=moment,
            actual=at,
            source_index=index,
            camera=alignment
            @ np.load(args.recording / "source" / sources[index]["file"])["camera"],
        )
    (args.out / "display.json").write_text(
        json.dumps(
            dict(
                arm=args.arm,
                settings=settings,
                GL=gl.info["GL_RENDERER"],
                eye_size=args.eye_size,
                recording=args.record,
                frames=len(costs),
                missed=misses,
                longest_consecutive=longest,
                completion_ms=dict(
                    zip(
                        ("p50", "p95", "p99", "p99.9"),
                        np.percentile(costs, [50, 95, 99, 99.9]).tolist(),
                        strict=True,
                    )
                ),
                service_ms=dict(
                    zip(
                        ("p50", "p95", "p99", "p99.9"),
                        np.percentile(service, [50, 95, 99, 99.9]).tolist(),
                        strict=True,
                    )
                ),
                max_ms=max(costs),
            ),
            indent=2,
        )
    )
    print((args.out / "display.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
