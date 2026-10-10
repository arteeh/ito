"""SLAM keyframe recall (#24), without CUDA: sector keyframes outlive map restarts, and
relocalization picks the one nearest the robot's gaze and says whether it switched.

uv run python e2e/slam_recall.py

The tracking itself needs MASt3R on CUDA; e2e/slam_replay.py on Ceres measures that.
"""

import math
from types import SimpleNamespace

import numpy as np

from ito.reconstruction import slam


def facing(degrees):
    """World-from-camera pose looking `degrees` left of the startup heading."""
    pose = np.eye(4)
    pose[:3, :3] = slam.turn(math.radians(degrees))
    return pose


def backend():
    b = slam.SLAMBackend.__new__(slam.SLAMBackend)
    b.frames = slam.Keyframes()
    b.spread = {}
    b.tracker = SimpleNamespace(reset_idx_f2k=lambda: None)
    b.placed = lambda transform: (facing(transform), 1.0)  # keyframe T_WC = its heading
    return b


def keyframe(degrees):
    return SimpleNamespace(T_WC=degrees)


def check(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def main():
    check(slam.RECALL, "recall on by default")
    check(abs(slam.heading(facing(90)[:3, :3]) - math.radians(90)) < 1e-9, "heading convention")
    b = backend()
    start = keyframe(0)
    b.start_map(start, facing(0), facing(0))
    # A +90 look outruns the map and restarts it there: the start's keyframe must survive.
    b.start_map(keyframe(90), facing(90), facing(90))
    check(start in b.spread.values(), "a restart forgot the start heading's keyframe")
    check(len(b.spread) == 2, b.spread)
    # Looking back at the start: relocalization switches to the start's keyframe.
    check(b.relocalize(facing(5)), "did not switch back to the start keyframe")
    check(list(b.frames) == [start], b.frames)
    # Already on the nearest keyframe: nothing to switch to, so no retry.
    check(not b.relocalize(facing(-3)), "switched with nothing nearer")
    # Gaze between sectors and the newest keyframe within half a sector: stay.
    b.frames[:] = [keyframe(80)]
    check(not b.relocalize(facing(85)), "left a keyframe already facing the gaze")
    # Old behaviour on request: restarts forget.
    slam.RECALL = False
    b = backend()
    b.start_map(keyframe(0), facing(0), facing(0))
    b.start_map(keyframe(90), facing(90), facing(90))
    check(len(b.spread) == 1, "ITO_SLAM_RECALL=0 must clear the sectors on restart")
    print("PASS slam_recall: sector keyframes outlive restarts; relocalize reports a switch")


if __name__ == "__main__":
    main()
