"""Shared-load stereo trace, using frozen step-1 maps and recorded pilot head motion.

Run via e2e/gpu_budget.sh; output stays outside the checkout.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_gl import View


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = [p for p in (root / "ito").rglob("*") if p.suffix in (".py", ".glsl")]
    paths += [
        root / "e2e" / name
        for name in (
            "gpu_budget.py",
            "gpu_training.py",
            "gpu_memory.ps1",
            "gpu_memory.py",
            "gpu_cupti.py",
            "gpu_trace.py",
        )
    ]
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def precise_unix():
    import ctypes

    value = ctypes.c_uint64()
    ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime(ctypes.byref(value))
    return (value.value - 116444736000000000) / 10_000_000


def load_worker(args, ready, start, stop, admission):
    from PIL import Image
    from slam_replay import motion_rows

    from ito.protocol import Intrinsics, RobotDescription
    from ito.reconstruction import mast3r_runtime
    from ito.reconstruction.slam import SLAMBackend

    mast3r_runtime.MODELS = args.work / "pp/models"
    recording = args.work / f"rf-run/out/{args.sequence}/rec"
    intrinsics = Intrinsics.model_validate_json((recording / "camera.json").read_text())
    robot = RobotDescription.model_validate_json((recording / "robot.json").read_text())
    rows = [json.loads(s) for s in (recording / "frames.jsonl").read_text().splitlines()]
    motion = [json.loads(s) for s in (recording / "motion.jsonl").read_text().splitlines()]
    samples = motion_rows(motion)
    arrivals = np.array([m["local"] for m in motion])
    backend = SLAMBackend(
        524288,
        intrinsics,
        report=lambda s: None,
        motion_noise=robot.motion_noise,
        frame_delay=robot.cameras[0].capture_delay,
        exposure=robot.cameras[0].exposure,
        voxel_size=0.04,
        fade_seconds=0.5,
    )
    if args.admit:
        backend.bind_admission(admission)
    import torch

    costs = []
    cursor = 0
    origin = rows[0]["captured"]
    for index in range(0, len(rows), 6):
        if stop.is_set():
            break
        row = rows[index]
        rgb = np.asarray(Image.open(recording / row["name"]).convert("RGB"))
        end = int(np.searchsorted(arrivals, row["captured"], side="right"))
        backend.add_motion(samples[cursor:end])
        cursor = end
        began = time.perf_counter()
        profiler = None
        map_trace = None
        if args.profile and index == args.warm_prefix + 6:
            from gpu_cupti import Cupti
            from gpu_trace import MapTrace

            profiler = Cupti()
            map_trace = MapTrace()
            profiler.__enter__()
            map_scope = map_trace.record(backend)
            map_scope.__enter__()
        try:
            with admission.integration():
                backend.integrate(
                    rgb,
                    None,
                    np.array(row["camera"], np.float32).reshape(4, 4),
                    row["captured"] - origin,
                    captured=row["captured"],
                )
        except InterruptedError:
            if admission.stopped.value:
                break
            raise
        finally:
            if profiler:
                map_scope.__exit__(None, None, None)
                profiler.__exit__(None, None, None)
                report = profiler.report()
                report["host_scopes"] = map_trace.report()
                (args.out / "cuda-trace.json").write_text(json.dumps(report, indent=2))
        torch.cuda.synchronize()
        costs.append((time.perf_counter() - began) * 1000)
        if torch.cuda.memory_reserved() >= 7 * 1024**3:
            raise RuntimeError("Tracking reservation exceeds 7 GiB; stopping before 8 GiB")
        if index == args.warm_prefix:
            warm_count = len(costs)
            ready.set()
            if not start.wait(90):
                raise RuntimeError("Display did not start")
    (args.out / "load.json").write_text(
        json.dumps(
            dict(
                kind="production SLAMBackend, every sixth recorded frame, no duty sleeps",
                warm_integrations=warm_count,
                integrated=len(costs) - warm_count,
                integration_ms=costs,
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                admission_chunks=admission.chunks.value,
                admission_overruns=admission.overruns.value,
                cuda_pool_peak_bytes=admission.memory_peak.value,
            ),
            indent=2,
        )
    )
    ready.set()


class GrantSender:
    def __init__(self, process):
        self.process = process
        self.queue = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def send(self, deadline):
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        self.queue.put_nowait(deadline)

    def run(self):
        while not self.stopped.is_set():
            try:
                deadline = self.queue.get(timeout=0.1)
                os.write(self.process.stdin.fileno(), f"{deadline:.9f}\n".encode("ascii"))
            except queue.Empty:
                continue
            except OSError as exc:
                if not self.stopped.is_set():
                    self.error = str(exc)
                return

    def close(self):
        self.stopped.set()
        self.process.stdin.close()
        self.thread.join(2)


class Residency:
    def __init__(self, pids):
        self.samples = []
        self.owned = []
        self.coverage = []
        self.query_ms = []
        self.pids = pids
        self.error = None
        self.stopped = threading.Event()
        command = [
            "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(Path(__file__).with_name("gpu_memory.ps1")),
            "-Owned",
            ",".join(str(pid) for pid in pids),
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        from gpu_memory import device_memory

        try:
            with device_memory() as used_mib:
                while not self.stopped.is_set():
                    began = time.perf_counter()
                    value = used_mib()
                    self.query_ms.append((time.perf_counter() - began) * 1000)
                    self.samples.append((time.perf_counter(), value))
                    owned = self.process.stdout.readline()
                    if not owned:
                        if self.stopped.is_set():
                            return
                        raise RuntimeError(self.process.stderr.read())
                    sample = json.loads(owned)
                    self.coverage.append(sample)
                    if sample["valid"] and all(pid in sample["found"] for pid in sample["targets"]):
                        self.owned.append((time.perf_counter(), sample["bytes"]))
                    self.stopped.wait(0.5)
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.stopped.set()
        self.process.terminate()
        self.thread.join(6)
        self.process.wait(timeout=3)


def main():
    from truth import align

    from ito.protocol import Intrinsics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence", choices=["sweep180", "tour"], default="tour")
    parser.add_argument("--load", choices=["none", "slam", "training", "both"], default="none")
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--occupied", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--admit", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--updates", action="store_true")
    parser.add_argument("--packets", type=int, default=2)
    parser.add_argument("--warm-prefix", type=int, default=12)
    parser.add_argument("--eye-size", help="WIDTHxHEIGHT; source calibration otherwise")
    parser.add_argument("--eye-fov", type=float, default=90)
    parser.add_argument("--target-format", choices=["f1", "f4"], default="f4")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--head-offset", type=float, default=0)
    parser.add_argument("--view-index", type=int)
    parser.add_argument("--linux-work")
    parser.add_argument("--distro")
    args = parser.parse_args()
    if args.frames < 1 or args.occupied < 0 or args.capacity < 0 or args.warm_prefix % 6:
        parser.error("Positive frame count, nonnegative sizes and six-frame warm prefix required")
    args.out.mkdir(parents=True, exist_ok=True)
    source_fingerprint = fingerprint()
    index = 692 if args.sequence == "sweep180" else 2283
    snap = np.load(args.work / f"m1-out/verified/repeat-1/{args.sequence}/snapshot-{index:06d}.npz")
    intrinsics = Intrinsics.model_validate_json(
        (args.work / f"rf-run/out/{args.sequence}/rec/camera.json").read_text()
    )
    truths = np.load(args.work / f"gs/out/{args.sequence}/data.npz")["poses"]
    world = align(snap["first_pose"], truths[int(snap["first_index"])])
    camera = np.asarray(
        world @ truths[args.view_index if args.view_index is not None else index], np.float32
    )
    records = snap["records"]
    if args.occupied:
        chosen = (
            np.linspace(0, len(records) - 1, args.occupied, dtype=int)
            if args.occupied <= len(records)
            else np.arange(args.occupied) % len(records)
        )
        records = records[chosen]
    occupied = len(records)
    if args.capacity and args.capacity < occupied:
        parser.error("Allocated capacity must hold the occupied records")
    if args.capacity and not args.incremental:
        records = np.pad(records, ((0, args.capacity - occupied), (0, 0), (0, 0)))
        records[occupied:, 1, :3] = 1
        records[occupied:, 2, 0] = 1
    from scipy.spatial.transform import Rotation, Slerp

    pilot_rows = [
        json.loads(s)
        for s in (args.work / f"rf-run/out/{args.sequence}/metrics.jsonl").read_text().splitlines()
    ]
    stamps = np.array([r["time"] for r in pilot_rows])
    pilot_heads = np.array([r["head"] for r in pilot_rows])
    duration = stamps[-1] - stamps[0]
    times = (np.arange(args.frames + 90) / 90 + args.head_offset) % (2 * duration)
    times = stamps[0] + np.minimum(times, 2 * duration - times)
    rotations = Slerp(stamps, Rotation.from_matrix(pilot_heads[:, :3, :3]))
    head_rotations = rotations(times).as_matrix()
    head_positions = np.stack(
        [np.interp(times, stamps, pilot_heads[:, k, 3]) for k in range(3)], -1
    )
    camera[:3, :3] = np.eye(3)  # The product anchor carries position; the pilot supplies rotation.
    context = mp.get_context("spawn")
    ready, start, stop = (context.Event() for _ in range(3))
    from ito.reconstruction.admission import Admission

    stopped = context.RawValue("b", False)
    admission = Admission(context, stopped)
    worker = training = grants = residency = view = None
    try:
        if args.load in ("slam", "both"):
            worker = context.Process(target=load_worker, args=(args, ready, start, stop, admission))
            worker.start()
            until = time.perf_counter() + 180
            while not ready.wait(0.1):
                if not worker.is_alive() or time.perf_counter() > until:
                    stop.set()
                    worker.join(2)
                    if worker.is_alive():
                        worker.kill()
                    raise RuntimeError("Tracking failed to initialize; see load log")
        if args.load in ("training", "both"):
            for name in ("ready", "start", "stop"):
                (args.out / name).unlink(missing_ok=True)
            linux_out = subprocess.check_output(
                [
                    "C:/Windows/System32/wsl.exe",
                    "-d",
                    args.distro,
                    "--",
                    "wslpath",
                    "-u",
                    args.out.as_posix(),
                ],
                text=True,
            ).strip()
            training = subprocess.Popen(
                [
                    "C:/Windows/System32/wsl.exe",
                    "-d",
                    args.distro,
                    "--",
                    args.linux_work + "/gs/.venv-gs/bin/python",
                    args.linux_work + "/m2/e2e/gpu_training.py",
                    "--work",
                    args.linux_work,
                    "--out",
                    linux_out,
                    "--sequence",
                    args.sequence,
                ]
                + (["--admit"] if args.admit else []),
                stdin=subprocess.PIPE,
                bufsize=0,
            )
            until = time.perf_counter() + 180
            while not (args.out / "ready").exists():
                if training.poll() is not None or time.perf_counter() > until:
                    (args.out / "stop").touch()
                    raise RuntimeError("Training failed to initialize")
                time.sleep(0.1)
        grants = GrantSender(training) if training is not None and args.admit else None
        residency = Residency([os.getpid()] + ([worker.pid] if worker else []))
        eye_intrinsics = intrinsics
        if args.eye_size:
            width, height = (int(x) for x in args.eye_size.split("x"))
            focal = np.tan(np.radians(args.eye_fov) / 2)
            eye_intrinsics = intrinsics.model_copy(
                update=dict(
                    width=width,
                    height=height,
                    fx=width / (2 * focal),
                    fy=height / (2 * focal),
                    cx=width / 2,
                    cy=height / 2,
                )
            )
        view = View(eye_intrinsics)
        if args.target_format != "f4":
            with view.context:
                view.target.release()
                view.target = view.context.simple_framebuffer(
                    view.size, components=4, dtype=args.target_format
                )
        publication = {"publish_ms": [], "poll_ms": [], "apply_ms": []}
        ring = None
        cursor = 0
        if args.incremental or args.updates:
            from ito import clock
            from ito.reconstruction.ring import UpdateRing

            ring = UpdateRing(context, clock.now(), 0.5)
            records = records.copy()
            records[:, 1, 3] = 1e30
            records[:, 3, 3] = 0

            pending_packet = None

            def packet(initial=False):
                nonlocal cursor, pending_packet
                if pending_packet is not None:
                    if not view.renderer.apply(pending_packet):
                        return False
                    cursor = (int(pending_packet.indices[-1]) + 1) % len(records)
                    pending_packet = None
                    return True
                end = min(cursor + ring.batch, len(records))
                indices = np.arange(cursor, end, dtype=np.uint32)
                began = time.perf_counter()
                assert ring.publish(
                    indices,
                    records[cursor:end],
                    max(args.capacity, len(records)),
                    occupied,
                    ring.write_position,
                    ring.epoch,
                )
                published = time.perf_counter()
                update = ring.poll()
                polled = time.perf_counter()
                while not view.renderer.apply(update):
                    if not initial:
                        pending_packet = update
                        return False
                    view.context.finish()
                applied = time.perf_counter()
                publication["publish_ms"].append((published - began) * 1000)
                publication["poll_ms"].append((polled - published) * 1000)
                publication["apply_ms"].append((applied - polled) * 1000)
                cursor = end % len(records)
                return True

            with view.context:
                began = time.perf_counter()
                for _ in range((len(records) + ring.batch - 1) // ring.batch):
                    packet(initial=True)
                view.context.finish()
                view.upload_ms.append((time.perf_counter() - began) * 1000)
        else:
            view.upload(records)
        (args.out / "memory-pids.json").write_text(json.dumps(residency.pids))
        durations, lateness = [], []
        from ito import clock

        with view.context, clock.display_priority() as scheduling:
            right = view.context.simple_framebuffer(
                view.size, components=4, dtype=args.target_format
            )

            def stereo(n):
                head = np.eye(4, dtype=np.float32)
                head[:3, :3] = head_rotations[n]
                head[:3, 3] = head_positions[n]
                view.renderer.sort(camera, head)
                for side, target in ((-1, view.target), (1, right)):
                    eye = head.copy()
                    eye[:3, 3] += head[:3, 0] * side * 0.032
                    view.renderer.draw(
                        camera, eye, view.projection, target, clear=(0, 0, 0, 0), sort=False
                    )
                view.context.finish()

            def pixels():
                return np.concatenate(
                    [
                        np.frombuffer(t.read(components=3, dtype="f1"), np.uint8)
                        .reshape(view.size[1], view.size[0], 3)[::-1]
                        .copy()
                        for t in (view.target, right)
                    ],
                    axis=1,
                )

            reference_hashes, reference_images, loaded_images = [], [], []
            selected = set(np.linspace(0, args.frames - 1, 8, dtype=int))
            if args.visual:
                for n in range(args.frames + 90):
                    stereo(n)
                    rgb = pixels()
                    reference_hashes.append(hashlib.sha256(rgb).hexdigest())
                    if n >= 90 and n - 90 in selected:
                        reference_images.append(rgb)
            admission.enabled.value = args.admit
            start.set()
            (args.out / "start").touch()
            epoch = time.perf_counter()
            # Replay recorded orientations at 90 Hz, camera position held at the frozen map.
            for n in range(args.frames + 90):
                if residency.error:
                    raise RuntimeError("GPU residency monitor failed: " + residency.error)
                if residency.owned and max(v for _, v in residency.owned) >= 8_000_000_000:
                    raise RuntimeError("Ito dedicated allocation reached 8 GB; stopping own jobs")
                dwm = [s["dwm_bytes"] for s in residency.coverage if s["dwm_bytes"] > 0]
                if (
                    dwm
                    and residency.samples
                    and (max(v for _, v in residency.samples) * 1024**2 - min(dwm) >= 8_000_000_000)
                ):
                    raise RuntimeError("Conservative Ito residency reached 8 GB; stopping own jobs")
                if residency.samples and residency.samples[-1][1] > 11_500:
                    raise RuntimeError("Device has under 0.8 GiB free; stopping own jobs")
                due = epoch + n / 90
                remaining = due - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                began = time.perf_counter()
                if args.updates:
                    for _ in range(args.packets):
                        if not packet():
                            break
                stereo(n)
                done = time.perf_counter()
                if args.admit:
                    training_turn = training is not None and (worker is None or n % 2 == 0)
                    admission.completed(due, 1 / 90)
                    if training is not None:
                        # The separate stress optimizer still uses expiring grants.
                        # Production reconstruction uses render feedback concurrently.
                        deadline = due + 1 / 90 - 0.002
                        grant = (
                            precise_unix() + deadline - time.perf_counter() if training_turn else 0
                        )
                        grants.send(grant)
                        if grants.error:
                            raise RuntimeError("Training grant pipe failed: " + grants.error)
                if args.visual:
                    rgb = pixels()
                    if hashlib.sha256(rgb).hexdigest() != reference_hashes[n]:
                        raise RuntimeError(f"Frozen image changed under load at frame {n}")
                    if n >= 90 and n - 90 in selected:
                        loaded_images.append(rgb)
                if n >= 90:
                    durations.append((done - began) * 1000)
                    lateness.append((done - due) * 1000)
            right.release()
        if args.visual:
            from PIL import Image, ImageDraw

            rows = []
            for reference, loaded in zip(reference_images, loaded_images, strict=True):
                rows.append(np.concatenate((reference, loaded), axis=1))
            sheet = Image.fromarray(np.concatenate(rows))
            sheet.thumbnail((1440, 2400))
            ImageDraw.Draw(sheet).text(
                (8, 8),
                "Before load (two eyes) | During load (two eyes)",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            sheet.save(args.out / "frozen-load.jpg", quality=85)
            (args.out / "visual.json").write_text(
                json.dumps(
                    dict(
                        stereo_frames=args.frames + 90,
                        startup_frames=90,
                        differing_frames=0,
                        scope="all stereo RGB bytes compared at identical recorded pilot poses; "
                        "readback run is excluded from deadline acceptance",
                    ),
                    indent=2,
                )
            )
        streak = longest = 0
        for duration in lateness:
            streak = streak + 1 if duration > 1000 / 90 else 0
            longest = max(streak, longest)
        np.savez(args.out / "frames.npz", duration_ms=durations, completion_ms=lateness)
        (args.out / "memory-coverage.json").write_text(json.dumps(residency.coverage, indent=2))
        if not residency.owned:
            raise RuntimeError("No complete WDDM memory sample; see memory-coverage.json")
        if fingerprint() != source_fingerprint:
            raise RuntimeError("Source changed during measurement; discard this run")
        report = dict(
            source_fingerprint=source_fingerprint,
            arguments={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            scope=(
                "diagnostic frozen-image comparison with readback"
                if args.visual
                else "90 Hz offscreen stereo GL completion; no compositor or scanout"
            ),
            gpu_memory_scope="persistent NVML total device use, including non-Ito processes",
            gpu_memory_query_ms={
                f"p{p:g}": float(np.percentile(residency.query_ms, p)) for p in (50, 99, 100)
            },
            gpu_memory_peak_mib=max(v for _, v in residency.samples),
            gpu_memory_samples=residency.samples,
            ito_dedicated_peak_bytes=max(v for _, v in residency.owned),
            ito_dedicated_samples=residency.owned,
            ito_counter_coverage=residency.coverage,
            ito_dedicated_scope="WDDM Dedicated Usage for owned Windows PIDs only",
            ito_residency_upper_bound_bytes=(
                max(v for _, v in residency.samples) * 1024**2
                - min(s["dwm_bytes"] for s in residency.coverage if s["dwm_bytes"] > 0)
            ),
            ito_residency_scope=(
                "maximum total device use minus minimum sampled DWM dedication on the same GPU; "
                "includes all Linux CUDA, driver memory and other apps; conservative offscreen "
                "Ito upper bound, not just Torch reservations"
            ),
            eye_size=view.size,
            eye_projection=view.projection.tolist(),
            target_format=args.target_format,
            admission=args.admit,
            trajectory="recorded pilot metrics head poses, Slerp/linear interpolation at 90 Hz",
            sequence=args.sequence,
            load=args.load,
            renderer=view.device,
            scheduling=scheduling,
            occupied=occupied,
            allocated=view.renderer.slot_capacity,
            sorted_slot_extent=view.renderer.draw_count,
            publication=publication,
            incremental=args.incremental,
            updates=args.updates,
            packets_per_frame=args.packets if args.updates else 0,
            sort_capacity=view.renderer.capacity,
            upload_ms=view.upload_ms,
            frames=args.frames,
            stereo_ms={f"p{p:g}": float(np.percentile(durations, p)) for p in (50, 95, 99, 99.9)},
            deadline_ms={f"p{p:g}": float(np.percentile(lateness, p)) for p in (50, 95, 99, 99.9)},
            longest_stall_ms=max(durations),
            longest_deadline_ms=max(lateness),
            missed_fraction=float(np.mean(np.array(lateness) > 1000 / 90)),
            longest_miss_run=longest,
        )
        np.savez(args.out / "frames.npz", duration_ms=durations, completion_ms=lateness)
        (args.out / "display.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in (
                        "publication",
                        "gpu_memory_samples",
                        "ito_dedicated_samples",
                        "ito_counter_coverage",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if residency:
            residency.close()
        stop.set()
        stopped.value = True
        (args.out / "stop").touch()
        if grants:
            grants.close()
        cleanup_errors = []
        if training:
            try:
                training.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training.kill()
                training.wait()
                cleanup_errors.append("Training did not stop after its stop signal")
            if training.returncode:
                cleanup_errors.append(f"Training exited {training.returncode}")
        if worker:
            worker.join(45 if args.profile else 10)
            if worker.is_alive():
                worker.kill()
                worker.join()
            if worker.exitcode:
                cleanup_errors.append(f"Tracking exited {worker.exitcode}")
        if view:
            view.close()
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
