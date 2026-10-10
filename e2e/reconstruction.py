"""Moving textured RGB-D room: DISPLAY=:97 uv run python e2e/reconstruction.py.

Runs for two minutes by default; --seconds controls the sustained portion.
Artifacts include screenshots, frame metrics and streaming/eviction measurements.
"""

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame

from ito import clock
from ito.desktop import DesktopState, DesktopWindow, PilotStatus
from ito.desktop.settings import load_budget
from ito.protocol import Intrinsics
from ito.reconstruction import Reconstruction
from ito.render import perspective, pose

OUTPUT = Path("e2e/out/reconstruction")
CAMERA = Intrinsics(width=96, height=72, fx=58, fy=58, cx=47.5, cy=35.5)
BUDGET = 4096


class Room:
    def __init__(self):
        y, x = np.mgrid[: CAMERA.height, : CAMERA.width]
        self.rays = np.stack(
            ((x - CAMERA.cx) / CAMERA.fx, -(y - CAMERA.cy) / CAMERA.fy, -np.ones_like(x)), axis=-1
        )

    def frame(self, camera):
        rays = self.rays @ camera[:3, :3].T
        origin = camera[:3, 3]
        # Rays start inside the room. Intersect the first of six enclosing walls.
        walls = np.where(rays >= 0, (4, 2, 9), (-4, -2, -9))
        with np.errstate(divide="ignore", invalid="ignore"):
            distances = (walls - origin) / rays
        depth = distances.min(axis=2)
        points = origin + rays * depth[..., None]
        checker = (np.floor(points * 3).sum(axis=2).astype(int) % 2)[..., None]
        colors = np.stack(
            (
                0.5 + 0.4 * np.sin(points[..., 2] * 0.7),
                0.5 + 0.4 * np.sin(points[..., 0] * 1.3),
                0.5 + 0.4 * np.cos(points[..., 1] * 2),
            ),
            axis=-1,
        )
        rgb = np.uint8(np.clip(colors * (0.5 + checker * 0.5) * 255, 0, 255))
        return rgb, depth.astype(np.float32)


class Observe:
    def __init__(self, source):
        self.source = source
        self.records = np.zeros((BUDGET, 4, 4), np.float32)
        self.revisions = 0
        self.counts = []
        self.evictions = 0
        self.refreshes = 0
        self.bytes = 0
        self.stall = False
        self.release_after = np.zeros(BUDGET)

    @property
    def max_splats(self):
        return self.source.max_splats

    def set_max_splats(self, value):
        self.source.set_max_splats(value)

    def poll(self):
        if self.stall:
            return None
        packet = self.source.poll()
        if packet is not None:
            assert packet.count <= BUDGET
            assert len(packet.indices) <= 4096
            previous = self.records[packet.indices]
            valid = (previous[:, 0, 3] > 0) & (packet.records[:, 0, 3] > 0)
            same = np.linalg.norm(previous[:, 0, :3] - packet.records[:, 0, :3], axis=1) < 0.18
            replaced = ((previous[:, 0, 3] > 0) & (packet.records[:, 0, 3] == 0)) | (valid & ~same)
            assert np.all(self.release_after[packet.indices[replaced]] <= clock.now()), (
                "Slot reused before its acknowledged fade finished"
            )
            pressure = packet.records[:, 3, 3] == 1
            self.release_after[packet.indices[pressure]] = clock.now() + packet.fade_seconds
            self.release_after[packet.indices[~pressure]] = 0
            self.refreshes += int(np.count_nonzero(valid & same))
            self.evictions += int(np.count_nonzero(packet.records[:, 3, 3] == 1))
            self.records[packet.indices] = packet.records
            self.counts.append(packet.count)
            self.bytes += packet.records.nbytes + packet.indices.nbytes
            self.revisions += 1
        return packet


def verify_setting(window, source, room, anchor):
    # Drive the real ImGui input and Apply button through SDL, then watch the worker shrink.
    ticks = 0
    rgb, depth = room.frame(anchor)

    def post(kind, **attributes):
        pygame.event.post(pygame.event.Event(kind, **attributes))

    def key(code, down):
        post(pygame.KEYDOWN if down else pygame.KEYUP, key=code)

    def drive(pilot):
        nonlocal ticks
        ticks += 1
        source.source.submit(rgb, depth, anchor)
        if ticks == 2:
            post(pygame.MOUSEMOTION, pos=(65, 150), rel=(0, 0), buttons=(0, 0, 0))
        elif ticks == 3:
            post(pygame.MOUSEBUTTONDOWN, pos=(65, 150), button=1)
        elif ticks == 4:
            post(pygame.MOUSEBUTTONUP, pos=(65, 150), button=1)
        elif ticks == 6:
            key(pygame.K_LCTRL, True)
            key(pygame.K_a, True)
        elif ticks == 7:
            key(pygame.K_a, False)
            key(pygame.K_LCTRL, False)
        elif ticks == 9:
            post(pygame.TEXTINPUT, text="2048")
        elif ticks == 11:
            key(pygame.K_RETURN, True)
        elif ticks == 12:
            key(pygame.K_RETURN, False)
            post(pygame.MOUSEMOTION, pos=(285, 150), rel=(0, 0), buttons=(0, 0, 0))
        elif ticks == 14:
            post(pygame.MOUSEBUTTONDOWN, pos=(285, 150), button=1)
        elif ticks == 15:
            post(pygame.MOUSEBUTTONUP, pos=(285, 150), button=1)
        elif ticks == 20:
            assert source.max_splats == 2048, (source.max_splats, window.overlay.max_splats)
            assert load_budget(BUDGET) == 2048, "Pilot setting was not saved"
        elif ticks == 80:
            assert window.renderer.count <= 2048, "Worker did not honor the reduced budget"
            assert not window.input.captured, "Editing the budget grabbed the pilot camera"
            assert np.allclose(pilot.head, np.eye(4)), "Editing the budget moved the pilot camera"

    window.run(source, state=lambda: DesktopState(anchor), on_input=drive, max_frames=85)
    print("PASS: ImGui budget editing, persistence, input capture and graceful budget reduction")


def verify_fade(window, source, anchor):
    """A stalled camera keeps its room; a smaller budget fades the excess out together."""
    target = window.context.simple_framebuffer((160, 120))
    energies = []
    projection = perspective(math.radians(70), 4 / 3)
    try:
        for step in range(20):
            if step == 5:
                source.set_max_splats(1)
            while (packet := source.poll()) is not None:
                window.renderer.apply(packet)
            window.renderer.draw(anchor, pose(), projection, target, clear=(0, 0, 0, 1))
            pixels = np.frombuffer(target.read(components=3), np.uint8)
            energies.append(int(pixels.sum()))
            time.sleep(0.2)
        assert energies[0] > 10000
        assert min(energies[:5]) > energies[0] * 0.9, "Stalled scene faded without pressure"
        assert energies[-1] < energies[0] * 0.01, "Budget pressure did not evict the scene"
        assert any(0.1 < value / energies[0] < 0.9 for value in energies[5:-1]), (
            "Eviction popped instead of fading through intermediate opacity"
        )
        assert window.context.error == "GL_NO_ERROR"
    finally:
        target.release()
    return energies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("capture-*.png"):
        old.unlink()
    os.environ["XDG_CONFIG_HOME"] = str(OUTPUT / "config")
    room = Room()
    anchor = pose((0, 0, -5))
    stop = threading.Event()
    failures = []
    samples = []
    snapshots = []
    began = clock.now()
    with Reconstruction(
        CAMERA, max_splats=BUDGET, voxel_size=0.12, fade_seconds=0.4
    ) as reconstruction:
        source = Observe(reconstruction)

        def produce():
            nonlocal anchor
            try:
                while not stop.is_set():
                    elapsed = clock.now() - began
                    angle = elapsed * 0.25
                    anchor = pose(
                        (2 * math.sin(angle), 0, 6 * math.cos(angle)), yaw=angle + math.pi
                    )
                    if elapsed < 2:
                        anchor = pose()
                    rgb, depth = room.frame(anchor)
                    reconstruction.submit(rgb, depth, anchor)
                    stop.wait(1 / 15)
            except Exception as exc:
                failures.append(exc)

        producer = threading.Thread(target=produce)
        producer.start()
        last_tick = clock.now()
        ticks = 0
        captures = set()

        def drive(pilot):
            nonlocal last_tick, ticks
            now = clock.now()
            elapsed = now - began
            samples.append((elapsed, now - last_tick))
            last_tick = now
            ticks += 1
            # Stall consumption, then let bounded packets catch up without a snapshot.
            source.stall = 4 < elapsed < 5
            for mark in (3, 8, max(10, args.seconds - 1)):
                if elapsed >= mark and mark not in captures:
                    captures.add(mark)
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F12))
                    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_F12))
                    records = source.records
                    live = records[:, 0, 3] > 0
                    deadlines = records[:, 1, 3]
                    live &= deadlines > now - reconstruction.epoch
                    assert np.count_nonzero(live) > BUDGET // 5, "Latest surroundings missing"
                    # Current pose must have newly observed surfaces in front of it.
                    local = (records[live, 0, :3] - anchor[:3, 3]) @ anchor[:3, :3]
                    visible = (local[:, 2] < -0.1) & (np.abs(local[:, 0]) < -local[:, 2])
                    assert visible.sum() > 100, "Scene did not follow the moving camera"
                    snapshots.append(int(visible.sum()))
            if elapsed > args.seconds:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

        try:
            with (
                (OUTPUT / "metrics.jsonl").open("w") as metrics,
                DesktopWindow((480, 360), fps=30, capture_dir=OUTPUT, max_splats=BUDGET) as window,
            ):
                window.run(
                    source,
                    state=lambda: DesktopState(
                        anchor, PilotStatus("CONNECTED", 0, "Procedural RGB-D room")
                    ),
                    on_input=drive,
                    metrics=metrics,
                )
                assert window.context.error == "GL_NO_ERROR"
                assert window.renderer.uploaded_bytes == source.bytes
                stop.set()
                producer.join(timeout=5)
                verify_setting(window, source, room, anchor)
                energies = verify_fade(window, source, anchor)
        finally:
            stop.set()
            producer.join(timeout=5)
        assert not failures, failures
        assert source.revisions > 40 and source.refreshes > BUDGET
        assert source.evictions > BUDGET, "Room movement never exercised budget eviction"
        assert max(source.counts) == BUDGET
        assert len(snapshots) >= 2
        captures = [
            pygame.surfarray.array3d(pygame.image.load(path))[:, 190:].astype(float)
            for path in sorted(OUTPUT.glob("capture-*.png"))
        ]
        assert len(captures) >= 2
        assert np.abs(captures[0] - captures[1]).mean() > 8, "Rendered surroundings did not change"
        times = np.array(samples)
        first = times[(times[:, 0] > 5) & (times[:, 0] < args.seconds / 2), 1]
        last = times[times[:, 0] > args.seconds * 0.75, 1]
        assert np.median(last) < np.median(first) * 1.4 + 0.005, "Frame time grew over the run"
        rows = [json.loads(line) for line in (OUTPUT / "metrics.jsonl").read_text().splitlines()]
        rendered = np.array(
            [(row["time"] - began, row["frame_ms"]) for row in rows if row["capture"] is None]
        )
        early_render = rendered[(rendered[:, 0] > 5) & (rendered[:, 0] < args.seconds / 2), 1]
        late_render = rendered[rendered[:, 0] > args.seconds * 0.75, 1]
        assert np.median(late_render) < np.median(early_render) * 1.6 + 4, (
            "Uncapped render work grew over the run"
        )
        report = dict(
            seconds=args.seconds,
            revisions=source.revisions,
            budget=BUDGET,
            max_count=max(source.counts),
            refreshes=source.refreshes,
            fading_evictions=source.evictions,
            uploaded_bytes=source.bytes,
            first_frame_ms=float(np.median(first) * 1000),
            last_frame_ms=float(np.median(last) * 1000),
            visible_samples=snapshots,
            fade_energy=energies,
            early_render_ms=float(np.median(early_render)),
            late_render_ms=float(np.median(late_render)),
        )
        (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    print("PASS: bounded live RGB-D stream, refresh, eviction, consumer stall, steady frame time")


if __name__ == "__main__":
    main()
