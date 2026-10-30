"""Play a tape into a candidate's Reconstruction worker exactly as the pilot feeds it, and film the
map the pilot would get. Started by tapevid.sh; runs on Windows with a build's Python:

    runtime\\python.exe -I replay.py TAPE OUT --code ROOT [--truth poses] [--fps 15] [--end S]

ROOT is the candidate's checkout: its ito package is imported instead of the build's.
The tape plays in real time: every frame is submitted when it was received (the worker drops what
it has no time for, as live), the robot's motion and sensor messages when they were published.
The splat packets the worker publishes are mirrored as the pilot's renderer keeps them (slots,
eviction, retire fades) and drawn with gsplat FPS times per tape second, from
  - path.avi      the tape's true camera path, smoothed (path.py): the same for every candidate
  - overview.avi  one fixed view over the room
OUT/ticks.jsonl has one line per drawn frame (tape time, the tape frame shown, splats, the
worker's published pose and status). --truth poses gives the worker the sim's exact pose of
every frame (branches with ITO_TRUTH support): the true-pose reference."""

import argparse
import inspect
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = sys.argv[sys.argv.index("--code") + 1] if "--code" in sys.argv else str(HERE.parents[1])
sys.path[:0] = [str(HERE), CODE]
if "--lib" in sys.argv:
    sys.path.append(sys.argv[sys.argv.index("--lib") + 1])

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from data import AXES, Recording, pose_matrix  # noqa: E402
from path import camera_path  # noqa: E402
from views import VIEWS, mujoco_camera, world_transform  # noqa: E402

WIDTH, HEIGHT = 960, 540
OVERVIEW = 6  # views.VIEWS: "room overview"
BACKGROUND = (0.04, 0.04, 0.06)


class Mirror:
    """The pilot renderer's copy of the worker's splat slots (ito/render/renderer.py apply),
    kept on the GPU."""

    def __init__(self, torch):
        self.torch = torch
        self.records = torch.zeros((0, 4, 4), device="cuda")
        self.live = torch.zeros(0, dtype=torch.bool, device="cuda")
        self.epoch = None
        self.fade = 0.0
        self.count = 0
        self.packets = 0

    def apply(self, update, now):
        torch = self.torch
        if update.capacity != len(self.records):
            records = torch.zeros((update.capacity, 4, 4), device="cuda")
            live = torch.zeros(update.capacity, dtype=torch.bool, device="cuda")
            n = min(len(self.records), update.capacity)
            records[:n], live[:n] = self.records[:n], self.live[:n]
            self.records, self.live = records, live
        if self.epoch is not None and update.epoch != self.epoch:
            self.records.zero_()
            self.live.zero_()
        self.epoch, self.fade, self.count = update.epoch, update.fade_seconds, update.count
        self.packets += 1
        if len(update.indices):
            records = update.records.copy()
            retiring = records[:, 3, 3] == 1
            records[retiring, 1, 3] = np.minimum(
                records[retiring, 1, 3], now - self.epoch + self.fade
            )
            index = torch.as_tensor(update.indices.astype(np.int64), device="cuda")
            values = torch.as_tensor(records, device="cuda")
            self.records[index] = values
            self.live[index] = values[:, 0, 3] > 0
        if update.acknowledge is not None:
            update.acknowledge()

    def drawable(self, now):
        r = self.records[self.live]
        if self.fade and self.epoch is not None and len(r):
            scene = now - self.epoch
            r[:, 0, 3] *= ((r[:, 1, 3] - scene) / self.fade).clamp(0, 1)
        return r


def draw(records, camera, K, torch, rasterization):
    if not len(records):
        return np.full((HEIGHT, WIDTH, 3), [int(255 * c) for c in BACKGROUND[::-1]], np.uint8)
    r = records
    view = torch.as_tensor(np.linalg.inv(camera @ AXES), device="cuda", dtype=torch.float32)
    k = torch.as_tensor(K, device="cuda", dtype=torch.float32)
    image, alpha, _ = rasterization(
        means=r[:, 0, :3],
        quats=torch.nn.functional.normalize(r[:, 2], dim=-1),
        scales=r[:, 1, :3],
        opacities=r[:, 0, 3].clamp(0, 1),
        colors=r[:, 3:4, :3],
        sh_degree=0,
        viewmats=view[None],
        Ks=k[None],
        width=WIDTH,
        height=HEIGHT,
        near_plane=0.02,
        far_plane=100,
    )
    background = torch.tensor(BACKGROUND, device="cuda")
    image = image[0] + (1 - alpha[0]) * background
    rgb = (image.clamp(0, 1) * 255).byte().cpu().numpy()
    return np.ascontiguousarray(rgb[..., ::-1])


def accepts(function, name):
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tape")
    p.add_argument("out")
    p.add_argument("--code")
    p.add_argument("--lib")
    p.add_argument("--truth", default="")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--end", type=float, help="stop after this many tape seconds")
    p.add_argument("--max-splats", type=int)
    a = p.parse_args()
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from gsplat import rasterization

    import ito
    from ito import clock
    from ito.protocol import Intrinsics, MotionNoise
    from ito.reconstruction import Reconstruction

    rec = Recording(a.tape)
    meta = rec.meta
    frames = rec.frames
    t0 = frames[0]["received"]
    end = frames[-1]["received"] - t0 if a.end is None else min(a.end, frames[-1]["received"] - t0)
    frames = [f for f in frames if f["received"] - t0 <= end]
    path_times, path_poses = camera_path(rec)
    world = world_transform(rec.truth_frames())
    overview_name, position, target = VIEWS[OVERVIEW]
    overview = world @ mujoco_camera(position, target)
    focal = WIDTH / 2 / math.tan(math.radians(90.0) / 2)
    K_overview = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]])
    K_path = rec.K_at((WIDTH, HEIGHT))

    settings = {}
    try:
        from ito.app.settings import Settings

        settings = Settings().model_dump()
    except Exception as exc:  # noqa: BLE001
        print("settings defaults unavailable:", exc)
    options = dict(
        max_splats=a.max_splats or settings.get("max_splats") or 1_000_000,
        backend="slam",
        tracking=settings.get("tracking", "robot"),
        motion_noise=MotionNoise(**meta["motion_noise"]),
        frame_delay=meta["frame_delay"],
        exposure=meta["exposure"],
        precision=settings.get("precision", "full"),
        input_resolution=settings.get("input_resolution", 100),
    )
    signature = inspect.signature(Reconstruction.__init__).parameters
    # Capture stamps: A1 tapes carry each frame's capture time (`captured`) and say whether it
    # is exact. Older tapes: a candidate that takes exact stamps (Camera.capture_exact, A1) gets
    # the sim's exposure time, as lane base's research replay does; older candidates get the
    # arrival stamp the pilot mapped frames by then, with the driver's frame delay.
    if any("captured" in f for f in frames):
        stamp = "captured"
        options["capture_exact"] = bool(meta.get("capture_exact", False))
    elif "capture_exact" in signature and any("exposed" in f for f in frames):
        stamp = "exposed"
        options.update(capture_exact=True, frame_delay=0.0)
    else:
        stamp = "received"
    options = {k: v for k, v in options.items() if k in signature}
    exposed = [f["received"] - f["exposed"] for f in frames if "exposed" in f]
    lag = float(np.median(exposed)) if exposed else 0.0  # for frames the sim gave no truth line

    def stamp_of(f):
        if stamp == "exposed":
            return f.get("exposed", f["received"] - lag)
        return f.get(stamp, f["received"])

    print("frames stamped by", stamp)
    print("ito from", Path(ito.__file__).parent, "options", {k: str(v) for k, v in options.items()})

    truth_modes = set(filter(None, a.truth.split(",")))
    worker = Reconstruction(Intrinsics(**meta["intrinsics"]), **options)
    give_truth = bool(truth_modes) and accepts(worker.submit, "truth")
    if truth_modes and not give_truth:
        raise SystemExit("this candidate's Reconstruction.submit takes no truth")

    # Every input as (tape time, kind, payload), in time order.
    times, poses, gyro, states = rec.motion
    events = [(t - t0, 1, (t, poses[i], gyro[i], states[i])) for i, t in enumerate(times)]
    for message in rec.sensors:
        stamps = [s["time"] for u in message["imu"] for s in u["samples"]]
        stamps += [g["time"] for g in message["ranges"]]
        if stamps:
            events.append((max(stamps) - t0, 2, message))
    events = [e for e in events if -5 <= e[0] <= end]
    events += [(f["received"] - t0, 0, f) for f in frames]
    ticks = int(end * a.fps)
    events += [(k / a.fps, 3, k) for k in range(ticks + 1)]
    events.sort(key=lambda e: (e[0], e[1]))

    # Frames are read and rectified ahead by a few threads (the pilot gets them decoded; one
    # thread reading PNGs over the WSL share falls behind real time).
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    readers = ThreadPoolExecutor(max_workers=6)
    pending = deque()
    upcoming = iter(frames)

    def refill():
        while len(pending) < 90:
            f = next(upcoming, None)
            if f is None:
                return
            pending.append(readers.submit(lambda f=f: (f["seq"], rec.rgb(f))))

    refill()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writers = {
        name: cv2.VideoWriter(str(out / f"{name}.avi"), fourcc, a.fps, (WIDTH, HEIGHT))
        for name in ("path", "overview")
    }
    # Encoding runs on a thread, off the feed's clock.
    encode = queue.Queue(maxsize=120)

    def encoder():
        while (item := encode.get()) is not None:
            writers[item[0]].write(item[1])

    encoding = threading.Thread(target=encoder, daemon=True)
    encoding.start()
    tick_log = (out / "ticks.jsonl").open("w")
    mirror = Mirror(torch)
    # The tape starts once the worker's loop runs (models loaded), so every candidate's film
    # starts at the same tape moment; the load time itself is not filmed.
    loading = clock.now()
    while clock.now() - loading < 300:
        worker.poll()
        if getattr(worker, "slot_capacity", None) is None or worker.slot_capacity.value > 0:
            break
        time.sleep(0.2)
    print(f"worker ready after {clock.now() - loading:.1f} s: {worker.status()}")
    start = clock.now() + 1.0  # tape time 0 on the pilot clock

    def local(tape_time):
        return start + tape_time

    def remote_to_local(driver_time):
        return local(driver_time - t0)

    submitted = accepted = late = ahead = 0
    last_frame = frames[0]["seq"]
    began = time.perf_counter()
    for when, kind, payload in events:
        wait = local(when) - clock.now()
        if wait > 0:
            time.sleep(wait)
        elif kind == 3 and wait < -0.5:
            late += 1
        if kind == 0:
            ahead = sum(future.done() for future in pending)
            seq, rgb = pending.popleft().result()
            refill()
            truth = None
            if give_truth:
                pose = payload["pose"]
                if pose is None:
                    continue
                exposed = payload.get("exposed", payload["received"])
                truth = (pose if "poses" in truth_modes else None, remote_to_local(exposed), None)
            kwargs = dict(measured=False)
            if truth is not None:
                kwargs["truth"] = truth
            submitted += 1
            if worker.submit(rgb, None, None, remote_to_local(stamp_of(payload)), **kwargs):
                accepted += 1
            last_frame = seq
        elif kind == 1:
            t, pose, spin, state = payload
            worker.motion([(remote_to_local(t), pose, spin, state)])
        elif kind == 2:
            imu = [
                dict(
                    sensor=u["sensor"],
                    mount=pose_matrix(u["mount"]["position"], u["mount"]["orientation"]),
                    gyro_sigma=u["gyro_sigma"],
                    accel_sigma=u["accel_sigma"],
                    samples=[
                        (
                            remote_to_local(s["time"]),
                            s["angular_velocity"],
                            s["specific_force"],
                            s["orientation"],
                        )
                        for s in u["samples"]
                    ],
                )
                for u in payload["imu"]
            ]
            ranges = [
                dict(
                    sensor=g["sensor"],
                    time=remote_to_local(g["time"]),
                    mount=pose_matrix(g["mount"]["position"], g["mount"]["orientation"]),
                    shape=(g["rows"], g["cols"]),
                    field_of_view=g["field_of_view"],
                    distances=[math.nan if d is None else d for d in g["distances"]],
                    sigma=g["sigma"],
                )
                for g in payload["ranges"]
            ]
            worker.sensors(imu, ranges)
        while True:
            update = worker.poll()
            if update is None:
                break
            mirror.apply(update, clock.now())
        if kind != 3:
            continue
        now = clock.now()
        records = mirror.drawable(now)
        i = min(np.searchsorted(path_times, when), len(path_times) - 1)
        with torch.no_grad():
            encode.put(("path", draw(records, path_poses[i], K_path, torch, rasterization)))
            encode.put(("overview", draw(records, overview, K_overview, torch, rasterization)))
        published = worker.pose()
        tick_log.write(
            json.dumps(
                dict(
                    tick=payload,
                    t=round(when, 3),
                    frame=int(last_frame),
                    splats=int((records[:, 0, 3] > 1 / 255).sum()) if len(records) else 0,
                    cost=round(clock.now() - now, 3),
                    ahead=ahead,
                    lag=round(now - local(when), 3),
                    pose=None if published is None else published[0].round(5).tolist(),
                    tracked=None if published is None else int(published[1]),
                    status=worker.status(),
                )
            )
            + "\n"
        )
    encode.put(None)
    encoding.join()
    for writer in writers.values():
        writer.release()
    tick_log.close()
    summary = dict(
        tape=a.tape,
        code=CODE,
        truth=sorted(truth_modes),
        seconds=round(end, 1),
        wall=round(time.perf_counter() - began, 1),
        submitted=submitted,
        accepted=accepted,
        late_ticks=late,
        packets=mirror.packets,
        splats=int(mirror.live.sum().item()),
        status=worker.status(),
        stamp=stamp,
        options={k: str(v) for k, v in options.items()},
        vram_peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
    )
    worker.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done", json.dumps(summary))


if __name__ == "__main__":
    main()
