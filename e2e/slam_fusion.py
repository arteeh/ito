"""SLAM splat fusion without CUDA: provisional splats, and repeat laps that do not double walls.

uv run python e2e/slam_fusion.py

MASt3R-SLAM needs CUDA, so a modelled tracker stands in for it: it walks the short slam_drive
route (forward, look +-90 degrees, back) for three laps in a box room with furniture,
matches frames less often the further the gaze has turned from the last match, recalls
sector keyframes, restarts after a second unmatched, and places views with a drifting
heading, position and depth scale. Its views go through SLAMBackend's own state and
splat code (`begin`, `place`, `expire`) exactly as the worker calls them: frames fused on
the robot's pose and frames of an unsettled map are provisional.

The same views also run through the old policy, where every placed view carves and
unmatched frames are dropped. Passes when the provisional path runs, a guessed view never
erases placed splats, a placed view replaces the guesses it sees, and after three laps the
scene holds under 1.7x the old policy's splats and off-surface splats, with laps 2-3 adding
under 2.5x as many. Fusing every guess and carving only on three agreeing views (the first
smooth-looks, 63eba18) held 3.0x, 3.1x and added 4.2x; guesses and views of an unsettled
map that may not carve are what the rest costs.
"""

import math
from types import SimpleNamespace

import numpy as np

from ito.reconstruction.rgbd import RGBDBackend
from ito.reconstruction.slam import SETTLE_FRAMES, SLAMBackend, turn

W, H = 256, 144
FX = W / 2 / math.tan(math.radians(31))
INTRINSICS = SimpleNamespace(width=W, height=H, fx=FX, fy=FX, cx=W / 2, cy=H / 2)
ROOM = np.array([[-2.0, 0.0, -3.0], [2.0, 2.4, 1.0]])
FURNITURE = [
    np.array([[-1.9, 0.0, -1.6], [-1.1, 0.7, -0.6]]),
    np.array([[0.9, 0, -2.9], [1.9, 0.9, -2.3]]),
]
y, x = np.mgrid[:H, :W]
RAYS = np.stack(((x - W / 2) / FX, -(y - H / 2) / FX, -np.ones((H, W))), -1).reshape(-1, 3)
STEP = 0.25  # seconds per processed frame, about the live SLAM rate


def pose(x, z, yaw):
    m = np.eye(4)
    m[:3, :3] = turn(yaw)
    m[:3, 3] = (x, 0.2, z)
    return m


def depth(camera):
    """Axial depth of the room and furniture per pixel."""
    d = RAYS @ camera[:3, :3].T
    o = camera[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (ROOM[1] - o) / d, (ROOM[0] - o) / d)
        t = np.nanmin(np.where(np.isfinite(t) & (t > 0), t, np.inf), axis=1)
        for box in FURNITURE:
            t1, t2 = (box[0] - o) / d, (box[1] - o) / d
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            t = np.where((near <= far) & (near > 0) & (near < t), near, t)
    return t


def off_surface(points):
    """Distance from each point to the nearest true surface."""
    distances = [np.minimum(np.abs(points - ROOM[0]), np.abs(points - ROOM[1])).min(1)]
    for box in FURNITURE:
        outside = np.linalg.norm(
            np.maximum(np.maximum(box[0] - points, points - box[1]), 0), axis=1
        )
        inside = np.all((points >= box[0]) & (points <= box[1]), axis=1)
        faces = np.minimum(np.abs(points - box[0]), np.abs(points - box[1])).min(1)
        distances.append(np.where(inside, faces, outside))
    return np.min(distances, axis=0)


def route(laps):
    """(time, x, z, gaze, gaze rate) per processed frame; None between laps."""
    t, z, gaze = 0.0, 0.0, 0.0

    def hold(seconds, speed=0.0):
        nonlocal t, z
        for _ in range(round(seconds / STEP)):
            t, z = t + STEP, z - speed * STEP
            yield t, 0.0, z, gaze, 0.0

    def look(degrees):
        nonlocal t, gaze
        target = math.radians(degrees)
        while abs(target - gaze) > 1e-9:
            step = math.copysign(min(abs(target - gaze), math.radians(45) * STEP), target - gaze)
            t, gaze = t + STEP, gaze + step
            yield t, 0.0, z, gaze, step / STEP

    for _ in range(laps):
        yield from hold(2)
        yield from hold(3, 0.38)
        yield from hold(1)
        yield from look(90)
        yield from hold(2)
        yield from look(-90)
        yield from hold(2)
        yield from look(0)
        yield from hold(1)
        yield from hold(3, -0.38)
        yield from hold(2)
        yield None


def matches(rng, turned):
    degrees = math.degrees(turned)
    return rng.random() < (
        0.85 if degrees < 10 else 0.5 if degrees < 20 else 0.2 if degrees < 30 else 0.05
    )


def views(seed, laps=3):
    """The modelled tracker's views: (time, true camera, placed camera, depth scale, kind),
    kind being "placed", "unsettled" (a map just restarted, recalled or correcting) or
    "guess" (unmatched, on the robot's pose); None between laps."""
    rng = np.random.default_rng(seed)
    keyframes = {}  # sector: (gaze, heading bias, scale)
    bias, scale, offset = 0.0, 1.0, np.zeros(3)
    last, unmatched_since, settled, correcting = None, None, 0, False
    for step in route(laps):
        if step is None:
            yield None
            continue
        now, x, z, gaze, rate = step
        robot = gaze - rate * 0.08 + rng.normal(0, math.radians(1))  # joint/IMU lag
        matched = last is None or matches(rng, abs(math.remainder(gaze - last, math.tau)))
        if not matched and keyframes:
            near = min(keyframes.values(), key=lambda k: abs(math.remainder(k[0] - gaze, math.tau)))
            if matches(rng, abs(math.remainder(near[0] - gaze, math.tau))) and rng.random() < 0.3:
                matched, settled = True, 0
                bias, scale = near[1] + rng.normal(0, math.radians(2)), near[2]
        if matched:
            unmatched_since, last = None, gaze
            bias += rng.normal(0, math.radians(1.5))
            offset = offset + rng.normal(0, 0.01, 3) * (1, 0.3, 1)
            stray = bias + gaze - robot
            if correcting or abs(stray) > math.radians(12):
                correcting = abs(stray) > math.radians(2)
                bias -= stray / 3 if correcting else 0
            keyframes[round(gaze / math.radians(30))] = (gaze, bias, scale)
            kind = "placed" if settled >= SETTLE_FRAMES and not correcting else "unsettled"
            settled += 1
            placed = pose(x, z, gaze + bias)
        else:
            unmatched_since = now if unmatched_since is None else unmatched_since
            placed = pose(x, z, robot)
            if now - unmatched_since < 1.0:
                kind = "guess"
            else:  # A fresh map at the robot's heading; its scale wanders.
                bias, scale = robot - gaze, scale * math.exp(rng.normal(0, 0.05))
                unmatched_since, last, settled, correcting = None, gaze, 1, False
                kind = "unsettled"
        placed[:3, 3] += offset
        # MASt3R's per-frame depth scale jitters, now and then badly.
        jitter = rng.normal(0, 0.03) if rng.random() > 0.05 else rng.normal(0, 0.2)
        yield now, pose(x, z, gaze), placed, scale * math.exp(jitter), kind


def backend():
    """A SLAMBackend without MASt3R: RGBDBackend's slots, then the tracking state, as
    SLAMBackend.__init__ sets them up."""
    b = SLAMBackend.__new__(SLAMBackend)
    RGBDBackend.__init__(b, 250_000, INTRINSICS, device="cpu")
    b.begin()
    return b


def fuse(b, true, placed, scale, now, provisional, rng):
    d = depth(true) * scale * (1 + rng.normal(0, 0.01, H * W))
    local = RAYS * d[:, None]
    points = local @ placed[:3, :3].T + placed[:3, 3]
    colors = np.full((len(points), 3), 128, np.uint8)
    sizes = d * 3 / FX  # three pixels, as SLAMBackend.fuse sizes splats
    b.place(
        points, colors, sizes, d.reshape(H, W), placed, (FX, FX, W / 2, H / 2), now, provisional
    )
    b.expire(now, np.inf)


def drive(seed, old):
    b, rng, laps = backend(), np.random.default_rng(seed + 1000), []
    for view in views(seed):
        if view is None:
            shown = (b.keys >= 0) & ~b.retiring
            off = off_surface(b.records[shown, 0, :3]) > 0.1
            laps.append(
                dict(
                    splats=int(shown.sum()),
                    off_surface=int(off.sum()),
                    provisional=int(b.provisional[shown].sum()),
                )
            )
            continue
        now, true, placed, scale, kind = view
        if old and kind == "guess":
            continue
        fuse(b, true, placed, scale, now, not old and kind != "placed", rng)
    return laps


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAIL slam_fusion: {message}")


def scenes():
    """Guessed and placed views of one wall, checked directly."""
    rng = np.random.default_rng(0)
    b = backend()
    fuse(b, pose(0, 0, 0), pose(0, 0, 0), 1.0, 1.0, False, rng)
    placed = int(np.count_nonzero(b.keys >= 0))
    # A guess 8 degrees off sees the wall at another place: it adds a provisional copy...
    fuse(b, pose(0, 0, 0), pose(0, 0, math.radians(8)), 1.0, 2.0, True, rng)
    live = (b.keys >= 0) & ~b.retiring
    check(
        np.count_nonzero(live & ~b.provisional) >= 0.97 * placed,
        "a guessed view erased placed splats",
    )
    first = int(np.count_nonzero(live & b.provisional))
    check(first > 0, "a guessed view added no provisional splats")
    # ...a second guess of the same view, 10 cm off the first, replaces it instead of adding
    # another copy...
    second = pose(0, 0, math.radians(8))
    second[0, 3] += 0.1
    fuse(b, pose(0, 0, 0), second, 1.0, 3.0, True, rng)
    b.expire(10.0, np.inf)
    kept = int(np.count_nonzero((b.keys >= 0) & b.provisional & (b.seen == 2.0)))
    check(kept < 0.2 * first, f"a second guess kept {kept} of the first one's {first} splats")
    # ...and a placed view where they were guessed puts the wall back in place.
    fuse(b, pose(0, 0, math.radians(8)), pose(0, 0, math.radians(8)), 1.0, 11.0, False, rng)
    b.expire(20.0, np.inf)
    left = int(np.count_nonzero((b.keys >= 0) & b.provisional))
    check(left < 0.1 * first, f"a placed view left {left} of the guesses it saw")
    check(b.provisional_frames == 0 and b.settled == 0, "SLAM tracking state was not set up")


def main():
    scenes()
    report = []
    for seed in range(1, 7):
        new, old = drive(seed, old=False), drive(seed, old=True)
        report.append((new, old))
        print(
            f"seed {seed} laps (splats/off-surface): now",
            [(lap["splats"], lap["off_surface"]) for lap in new],
            "old",
            [(lap["splats"], lap["off_surface"]) for lap in old],
        )
    mean = lambda f: float(np.mean([f(r) for r in report]))  # noqa: E731
    splats = mean(lambda r: r[0][-1]["splats"]) / mean(lambda r: r[1][-1]["splats"])
    off = mean(lambda r: r[0][-1]["off_surface"]) / mean(lambda r: r[1][-1]["off_surface"])
    growth = mean(lambda r: r[0][-1]["splats"] - r[0][0]["splats"])
    old_growth = mean(lambda r: r[1][-1]["splats"] - r[1][0]["splats"])
    summary = dict(
        lap3_splats_vs_old=round(splats, 2),
        lap3_off_surface_vs_old=round(off, 2),
        laps_2_3_added=round(growth),
        laps_2_3_added_old=round(old_growth),
        provisional_lap3=round(mean(lambda r: r[0][-1]["provisional"])),
    )
    print(summary)
    check(splats < 1.7, f"three laps hold {splats:.2f}x the old policy's splats")
    check(off < 1.7, f"three laps hold {off:.2f}x the old policy's off-surface splats")
    check(growth < 2.5 * old_growth, f"laps 2-3 added {growth:.0f} splats, old {old_growth:.0f}")
    print(
        "PASS slam_fusion: provisional splats fuse and give way; repeat laps do not double the room"
    )


if __name__ == "__main__":
    main()
