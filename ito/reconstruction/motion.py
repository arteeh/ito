"""The robot's own camera motion, on the pilot clock: where it was, and how sure it is.

Samples arrive from the link thread through MotionRing (shared memory, never blocking the
pilot) and the reconstruction worker keeps the last seconds of them in a MotionBuffer. Poses
are world-from-camera in Ito axes, world being the robot's odometry where it started.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .ring import MOTION_STATES as STATES

# A robot that is held or has fallen knows its camera's rotation, roughly, but not where
# its body went: its odometry factors loosen this much.
UNRELIABLE = 10.0


class MotionBuffer:
    """A time-ordered window of the robot's camera poses, with the robot's noise model."""

    def __init__(self, noise, seconds=10.0):
        self.noise = noise
        self.seconds = seconds
        self.times = np.empty(0)
        self.rotations = None
        self.positions = np.empty((0, 3))
        self.spins = np.empty((0, 3))
        self.states = np.empty(0, dtype=int)
        self.travel = np.empty(0)  # metres the camera moved since the window's first sample
        self.slerp = None

    def __len__(self):
        return len(self.times)

    def extend(self, rows):
        """Rows as MotionRing.read returns them; older or repeated times are ignored."""
        if not len(rows):
            return
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        last = self.times[-1] if len(self.times) else -math.inf
        keep = np.concatenate(([True], np.diff(rows[:, 0]) > 0)) & (rows[:, 0] > last)
        rows = rows[keep]
        if not len(rows):
            return
        poses = rows[:, 1:17].reshape(-1, 4, 4)
        rotations = Rotation.from_matrix(poses[:, :3, :3])
        times = np.concatenate((self.times, rows[:, 0]))
        positions = np.concatenate((self.positions, poses[:, :3, 3]))
        quaternions = rotations.as_quat()
        if self.rotations is not None:
            quaternions = np.concatenate((self.rotations.as_quat(), quaternions))
        spins = np.concatenate((self.spins, rows[:, 17:20]))
        states = np.concatenate((self.states, rows[:, 20].astype(int)))
        start = np.searchsorted(times, times[-1] - self.seconds)
        start = min(start, len(times) - 1)
        self.times, self.positions = times[start:], positions[start:]
        self.rotations = Rotation.from_quat(quaternions[start:])
        self.spins, self.states = spins[start:], states[start:]
        steps = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        self.travel = np.concatenate(([0.0], np.cumsum(steps)))
        self.slerp = Slerp(self.times, self.rotations) if len(self.times) > 1 else None

    def covers(self, t, slack=0.1):
        """Whether t is within the window, or at most `slack` seconds past its newest sample."""
        return len(self.times) > 0 and self.times[0] <= t <= self.times[-1] + slack

    def at(self, t):
        """World-from-camera at t: interpolated, carried on by the newest angular velocity
        for up to 0.1 s past the newest sample, held beyond either end."""
        t = float(t)
        result = np.eye(4)
        if t >= self.times[-1] or self.slerp is None:
            ahead = min(max(t - self.times[-1], 0.0), 0.1)
            turn = Rotation.from_rotvec(self.spins[-1] * ahead)
            result[:3, :3] = (self.rotations[-1] * turn).as_matrix()
            result[:3, 3] = self.positions[-1]
            return result
        t = max(t, self.times[0])
        result[:3, :3] = self.slerp([t]).as_matrix()[0]
        result[:3, 3] = [np.interp(t, self.times, self.positions[:, i]) for i in range(3)]
        return result

    def poses(self, times):
        """at() for many times at once: (Rotation of each, positions)."""
        times = np.asarray(times, dtype=float)
        last = self.times[-1]
        clamped = np.clip(times, self.times[0], last)
        if self.slerp is None:
            rotations = Rotation.from_quat(np.repeat(self.rotations.as_quat()[-1:], len(times), 0))
        else:
            rotations = self.slerp(clamped)
        positions = np.column_stack(
            [np.interp(clamped, self.times, self.positions[:, i]) for i in range(3)]
        )
        ahead = np.clip(times - last, 0.0, 0.1)
        if (ahead > 0).any():
            turned = self.rotations[-1] * Rotation.from_rotvec(self.spins[-1] * ahead[:, None])
            quats = rotations.as_quat()
            quats[ahead > 0] = turned.as_quat()[ahead > 0]
            rotations = Rotation.from_quat(quats)
        return rotations, positions

    def angular_velocity(self, t):
        """Camera-frame rad/s at t, from the sample nearest it."""
        index = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        if index and abs(self.times[index - 1] - t) < abs(self.times[index] - t):
            index -= 1
        return self.spins[index]

    def _timing(self, t):
        """Radians the camera's pose at t is off by from its stamp's error: speed x time_s."""
        return float(np.linalg.norm(self.angular_velocity(t))) * self.noise.time_s

    def _unreliable(self, t0, t1):
        lo, hi = np.searchsorted(self.times, (min(t0, t1), max(t0, t1)))
        states = self.states[max(lo - 1, 0) : hi + 1]
        return bool(np.isin(states, (STATES.index("held"), STATES.index("fallen"))).any())

    def between(self, t0, t1):
        """The camera's motion from t0 to t1 in the camera frame at t0, and its 6x6 covariance
        (rotation vector first, then translation)."""
        relative = np.linalg.inv(self.at(t0)) @ self.at(t1)
        dt = abs(t1 - t0)
        travelled = abs(
            np.interp(t1, self.times, self.travel) - np.interp(t0, self.times, self.travel)
        )
        noise = self.noise
        rotation = math.hypot(
            math.radians(math.hypot(noise.rotation_deg, noise.rotation_deg_per_s * dt)),
            self._timing(t0),
            self._timing(t1),
        )
        translation = noise.translation_m + noise.translation_fraction * travelled
        scale = UNRELIABLE if self._unreliable(t0, t1) else 1.0
        sigma = np.repeat((rotation * scale, translation * scale), 3)
        return relative, np.diag(sigma**2)

    def gravity_sigma(self, t):
        """Radians of pitch and roll error at t, one sigma."""
        scale = UNRELIABLE if self._unreliable(t, t) else 1.0
        return math.hypot(math.radians(self.noise.gravity_deg), self._timing(t)) * scale
