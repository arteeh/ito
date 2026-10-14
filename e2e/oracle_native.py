"""Native NVIDIA Windows GL with the real WSL optimizer, no readback or memory cap.

Under ceres.lock: RELEASE/runtime/python.exe -I e2e/oracle_native.py RECORDING
  --out OUT --linux-root WSL_CHECKOUT --arm main-TT|TT --load --seconds 20
Freezes the recorded mature GL map and applies two real upload packets per frame.
This isolates native renderer contention; it is not a live-map or headset acceptance run.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]


class Feedback:
    def __init__(self, path):
        self.path = path
        self.value = 0
        self.error = None
        self.wake = threading.Event()
        self.stop = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop:
            self.wake.wait(0.05)
            self.wake.clear()
            if not self.stop:
                try:
                    self.path.write_text(str(self.value))
                except OSError as exc:
                    self.error = str(exc)
                    return

    def send(self, value):
        self.value = value
        self.wake.set()

    def close(self):
        self.stop = True
        self.wake.set()
        self.thread.join(2)


def main():
    from measure_gl import View, projection

    from ito import clock
    from ito.protocol import Intrinsics
    from ito.reconstruction.admission import Admission
    from ito.reconstruction.ring import SplatUpdate

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--linux-root", required=True)
    p.add_argument("--arm", choices=("main-TT", "TT"), required=True)
    p.add_argument("--load", action="store_true")
    p.add_argument("--seconds", type=float, default=20)
    args = p.parse_args()
    assert os.name == "nt"
    args.out.mkdir(parents=True, exist_ok=True)
    sequence = args.recording.name
    moment = 78 if sequence == "tour" else 12
    snapshot = np.load(
        args.recording.parent / "visual" / f"{sequence}-{args.arm}" / f"snapshot-{moment:.1f}.npz"
    )
    records = snapshot["records"].copy()
    records[:, 1, 3] = 1e30
    records[:, 3, 3] = 0
    camera = snapshot["camera"].astype(np.float32)
    k = Intrinsics(width=2160, height=2160, fx=1080, fy=1080, cx=1080, cy=1080)
    view = View(k)
    renderer = view.renderer
    view.target.release()
    view.target = view.context.simple_framebuffer(view.size, components=4, dtype="f1")
    other = view.context.simple_framebuffer(view.size, components=4, dtype="f1")
    ctx = mp.get_context("spawn")
    admission = Admission(ctx, ctx.RawValue("b", False))
    count = int((records[:, 0, 3] > 0).sum())
    cursor = 0
    revision = 0

    def upload():
        nonlocal cursor, revision
        end = min(len(records), cursor + 4096)
        packet = SplatUpdate(
            np.arange(cursor, end, dtype=np.uint32),
            records[cursor:end],
            len(records),
            count,
            revision,
            0,
            1,
            0.5,
        )
        if renderer.apply(packet):
            cursor = end % len(records)
            revision += 1
            return True
        return False

    with view.context:
        for _ in range(math.ceil(len(records) / 4096)):
            while not upload():
                view.context.finish()
        view.context.finish()
    child = feedback = None
    try:
        if args.load:
            load = args.out / "load"
            load.mkdir(exist_ok=True)
            for name in ("ready", "start", "stop", "pause"):
                (load / name).unlink(missing_ok=True)
            linux_out = subprocess.check_output(
                ["wsl.exe", "--exec", "wslpath", "-u", str(load)], text=True
            ).strip()
            linux_rec = subprocess.check_output(
                ["wsl.exe", "--exec", "wslpath", "-u", str(args.recording)], text=True
            ).strip()
            child = subprocess.Popen(
                [
                    "wsl.exe",
                    "--exec",
                    "env",
                    "ITO_ORACLE_LOCK_HELD=1",
                    "bash",
                    args.linux_root + "/e2e/oracle.sh",
                    args.linux_root + "/e2e/oracle_native_load.py",
                    linux_rec,
                    "--out",
                    linux_out,
                    "--seconds",
                    str(args.seconds + 2),
                ],
                stdout=(args.out / "load.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            until = time.perf_counter() + 180
            while not (load / "ready").exists():
                if child.poll() is not None or time.perf_counter() > until:
                    raise RuntimeError("Native countercheck mapper failed to initialize")
                time.sleep(0.05)
            feedback = Feedback(load / "pause")
        costs, services = [], []
        misses = longest = consecutive = 0
        projection_matrix = projection(k)
        with view.context, clock.display_priority() as scheduling:
            # Prime shader/sort/allocator state, without training or measured deadlines.
            for _ in range(30):
                renderer.draw(camera, np.eye(4, dtype=np.float32), projection_matrix, view.target)
            view.context.finish()
            if args.load:
                (load / "start").write_text(str(time.time()))
            epoch = due = time.perf_counter()
            while (now := time.perf_counter() - epoch) < args.seconds:
                began = time.perf_counter()
                for _ in range(2):
                    if not upload():
                        break
                head = np.eye(4, dtype=np.float32)
                head[0, 3] = 0.03 * math.sin(now * 2)
                renderer.sort(camera, head)
                for side, target in ((-1, view.target), (1, other)):
                    eye = head.copy()
                    eye[0, 3] += side * 0.032
                    renderer.draw(
                        camera, eye, projection_matrix, target, sort=False, clear=(0, 0, 0, 0)
                    )
                view.context.finish()
                done = time.perf_counter()
                elapsed = done - due
                missed = max(0, math.ceil(elapsed * 90) - 1)
                misses += missed
                consecutive = consecutive + missed if missed else 0
                longest = max(longest, consecutive)
                costs.append(elapsed * 1000)
                services.append((done - began) * 1000)
                admission.completed(due, 1 / 90)
                if args.load:
                    remaining = admission.pause_until.value - time.perf_counter()
                    feedback.send(time.time() + remaining)
                    if feedback.error:
                        raise RuntimeError("Render feedback failed: " + feedback.error)
                    if child.poll() is not None:
                        raise RuntimeError("Mapper exited during native GL countercheck")
                due += max(1, missed + 1) / 90
                remaining = due - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        report = dict(
            renderer=view.device,
            sequence=sequence,
            arm=args.arm,
            optimizer=args.load,
            scope="native frozen map, controlled uploads, online optimizer; no compositor",
            scheduling=scheduling,
            settings=dict(precision="full", input_resolution=100, max_splats=1_000_000),
            seconds=args.seconds,
            eye_size=2160,
            occupied=count,
            capacity=len(records),
            frames=len(costs),
            missed=misses,
            longest=longest,
            completion_ms={f"p{p}": float(np.percentile(costs, p)) for p in (50, 95, 99, 99.9)},
            service_ms={f"p{p}": float(np.percentile(services, p)) for p in (50, 95, 99, 99.9)},
            max_ms=max(costs),
        )
        (args.out / "display.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        if feedback:
            feedback.close()
        if child:
            (load / "stop").touch()
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                # The owned bridge is the only process terminated; never use process-name kills.
                child.terminate()
                child.wait(10)
        other.release()
        view.close()


if __name__ == "__main__":
    main()
