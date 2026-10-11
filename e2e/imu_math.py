"""CPU unit checks requested for #33; optional recorded simulator and slam_replay poses.

uv run python e2e/imu_math.py [e2e/out/imu_prior/poses.npz] [RECORDING/frames.jsonl]
"""

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

from ito.reconstruction.prior import AXES, ROTATION_GATE, YAW_RATE, blend, predict, split_sim3, turn

RECORDINGS = [Path(p) for p in sys.argv[1:]]


class PoseMath(unittest.TestCase):
    def test_sim3(self):
        rng = np.random.default_rng(33)
        for _ in range(300):
            scale = np.exp(rng.uniform(-4, 4))
            rotation = turn(rng.uniform(-math.pi, math.pi))
            matrix = np.eye(4)
            matrix[:3, :3] = rotation * scale
            matrix[:3, 3] = rng.normal(size=3)
            rigid, actual = split_sim3(matrix)
            self.assertAlmostEqual(actual, scale)
            np.testing.assert_allclose(
                rigid[:3, :3], rotation * AXES[:, None] * AXES[None, :], atol=1e-12
            )
            np.testing.assert_allclose(rigid[:3, 3], matrix[:3, 3] * AXES)
        for bad in (
            np.diag([-1, 1, 1, 1]),
            np.zeros((4, 4)),
            np.full((4, 4), np.nan),
            np.diag([1, 2, 1, 1]),
        ):
            self.assertIsNone(split_sim3(bad)[0])

    def test_gravity_gate_and_yaw(self):
        prior = np.eye(4)
        visual = np.eye(4)
        visual[:3, :3] = turn(math.radians(4))
        visual[:3, 3] = [1, 2, 3]
        yaw = 0
        for _ in range(600):
            result, next_yaw = blend(prior, visual, yaw, 1 / 30)
            self.assertLessEqual(abs(next_yaw - yaw), YAW_RATE / 30)
            np.testing.assert_allclose(result[:3, 3], visual[:3, 3])
            np.testing.assert_allclose(result[:3, 1], [0, 1, 0])
            yaw = next_yaw
        self.assertGreater(yaw, math.radians(2))
        for axis in (0, 1, 2):
            angle = ROTATION_GATE + 0.01
            r = np.eye(3)
            a, b = [i for i in range(3) if i != axis]
            r[a, a] = r[b, b] = math.cos(angle)
            r[a, b], r[b, a] = -math.sin(angle), math.sin(angle)
            visual[:3, :3] = r
            result, bias = blend(prior, visual, 0, 0.1)
            self.assertIsNone(result)
            self.assertEqual(bias, 0)
        visual[:3, :3] = turn(math.radians(-179))
        prior[:3, :3] = turn(math.radians(179))
        self.assertIsNotNone(blend(prior, visual, 0, 0.1)[0])
        # A long gap cannot admit a large yaw step; rejected NaNs never enter the map.
        self.assertLessEqual(abs(blend(prior, visual, 0, 100)[1]), YAW_RATE * 0.25)
        visual[0, 0] = np.nan
        self.assertIsNone(blend(prior, visual, 0, 0.1)[0])

    def test_frame_rate_and_tilt(self):
        prior = np.eye(4)
        angle = math.radians(30)
        prior[:3, :3] = [
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)],
        ]
        visual = prior.copy()
        visual[:3, :3] = turn(math.radians(5)) @ prior[:3, :3]
        biases = []
        for rate in (10, 30, 90):
            yaw = 0
            for _ in range(rate * 5):
                fused, yaw = blend(prior, visual, yaw, 1 / rate)
                np.testing.assert_allclose(fused[1, :3], prior[1, :3], atol=1e-12)
            biases.append(yaw)
        np.testing.assert_allclose(biases, biases[0], atol=1e-12)

    def test_recorded(self):
        rng = np.random.default_rng(33)
        count = 0
        for path in RECORDINGS:
            if path.suffix == ".npz":
                with np.load(path) as recording:
                    poses = recording["prior"]
                    for prior, truth in zip(poses, recording["truth"], strict=True):
                        self.assertIsNotNone(blend(prior, truth, 0, 1 / 30)[0])
            else:
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                poses = [np.array(row["camera"]).reshape(4, 4) for row in rows if row.get("camera")]
            for pose in poses:
                count += 1
                visual = pose.copy()
                visual[:3, :3] = turn(math.radians(rng.normal(0, 1))) @ pose[:3, :3]
                fused, _ = blend(pose, visual, 0, 1 / 30)
                self.assertIsNotNone(fused)
                np.testing.assert_allclose(fused[1, :3], pose[1, :3], atol=1e-12)
                visual[:3, :3] = turn(math.radians(25)) @ pose[:3, :3]
                self.assertIsNone(blend(pose, visual, 0, 1 / 30)[0])
                # Metres in a prior cannot displace a monocular map of unknown scale.
                moved = pose.copy()
                moved[:3, 3] += 100
                np.testing.assert_array_equal(predict(moved, pose, 0)[:3, 3], pose[:3, 3])
        if RECORDINGS:
            self.assertGreater(count, 100)
            print(f"Checked {count} recorded poses with noise and false visual turns")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
