"""FrameJoin: every decoded frame reaches the pilot with its exposure's pose.

uv run python e2e/frame_join.py

No display, no network: a synthetic 30 fps stream whose pts drift a tick off the metadata,
lose metadata, and arrive in either order, plus a timing check of the hot path.
"""

import math
import random
import time
from fractions import Fraction
from types import SimpleNamespace

from ito.app.frames import TICKS, FrameJoin, slerp
from ito.protocol import Depth, FrameMetadata, Pose

STEP = 3003  # 30 fps in 90 kHz ticks


def yaw_quat(angle):
    return (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))


def meta(n, *, depth=None):
    key = n * STEP
    return FrameMetadata(
        camera="head",
        sequence=n,
        capture_time=100 + key / TICKS,
        video_pts=key,
        camera_pose=Pose(position=(n * 0.01, 0.0, 0.0), orientation=yaw_quat(n * 0.02)),
        head_angles=(n * 0.01, 0.0),
        body_yaw=n * 0.02,
        depth=depth,
    )


def frame(n, wobble=0):
    return SimpleNamespace(pts=n * STEP + wobble, time_base=Fraction(1, TICKS), n=n)


def check(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def run(events):
    join, out = FrameJoin(), []
    for kind, value in events:
        out += join.decoded(value) if kind == "v" else join.described(value)
    return join, out


def main():
    depth = Depth.from_bytes(2, 2, bytes(8))
    # 1. Off-by-one pts on almost half the frames, metadata first: all join exactly, with depth.
    events = []
    for n in range(200):
        events += [("m", meta(n, depth=depth)), ("v", frame(n, random.choice((-1, 0, 0, 1))))]
    join, out = run(events)
    check(len(out) == 200, f"one-tick drift joined {len(out)} of 200")
    check(join.joins["exact"] == 200, f"joins {join.joins}")
    check(all(m.depth is depth for _, m in out), "exact joins keep their depth")

    # 2. Every third metadata lost: those frames interpolate between their neighbours.
    events = []
    for n in range(90):
        if n % 3 != 1:
            events.append(("m", meta(n, depth=depth)))
        events.append(("v", frame(n)))
    events.append(("m", meta(90)))
    join, out = run(events)
    check(len(out) == 90, f"lost metadata joined {len(out)} of 90")
    check(join.joins["between"] == 30, f"joins {join.joins}")
    for f, m in out:
        want = meta(f.n)
        check(abs(m.capture_time - want.capture_time) < 1e-6, f"frame {f.n} capture time")
        check(abs(m.camera_pose.position[0] - want.camera_pose.position[0]) < 1e-9, "lerp")
        dot = sum(
            a * b
            for a, b in zip(m.camera_pose.orientation, want.camera_pose.orientation, strict=True)
        )
        check(abs(abs(dot) - 1) < 1e-9, f"slerp frame {f.n}")
        check(abs(m.body_yaw - want.body_yaw) < 1e-9, "body yaw")
        check(f.n % 3 != 1 or m.depth is None, "interpolated frames carry no depth")

    # 3. Video before its metadata: it waits for the metadata, then joins exactly.
    events = []
    for n in range(50):
        events += [("v", frame(n)), ("m", meta(n))]
    join, out = run(events)
    check(len(out) == 50 and join.joins["exact"] == 50, f"video-first joins {join.joins}")

    # 4. Metadata stops: frames still flow, nearest-pose then bare, in order, never dropped.
    events = [("m", meta(0)), ("v", frame(0))]
    for n in range(1, 30):
        events.append(("v", frame(n)))
    join, out = run(events)
    check(len(out) == 29, f"no-metadata frames reached {len(out)} of 29 (last waits)")
    check(join.joins["nearest"] == 1 and join.joins["bare"] == 27, f"joins {join.joins}")
    check(out[-1][1].camera_pose is None and out[-1][1].depth is None, "bare has no pose")
    times = [m.capture_time for _, m in out]
    check(times == sorted(times) and len(set(times)) == len(times), "capture order")

    # 5. Slerp takes the short way round across the quaternion sign flip.
    q = slerp((0, 0, 0, 1), (0, 0, 0, -1), 0.5)
    check(abs(abs(q[3]) - 1) < 1e-9, "slerp sign flip")

    # 6. Hot path: a join costs microseconds, not milliseconds.
    metas = [meta(n) for n in range(3000)]
    frames = [frame(n, n % 3 - 1) for n in range(3000)]
    join = FrameJoin()
    start = time.perf_counter()
    for m, f in zip(metas, frames, strict=True):
        join.described(m)
        join.decoded(f)
    per = (time.perf_counter() - start) / 3000 * 1e6
    check(per < 200, f"{per:.0f} us per frame")
    print(f"PASS frame_join ({per:.0f} us per frame incl. diagnostics stubs)")


if __name__ == "__main__":
    main()
