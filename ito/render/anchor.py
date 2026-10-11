"""Glide the anchor's position between updates, so the eye never steps at camera rate.

Anchors arrive at the camera or SLAM rate (2-30 Hz) while the display runs at 90-120 Hz.
Each display frame moves the position from where it was drawn last toward the newest
anchor over one update interval: about one interval of delay, and no steps. Rotation
stays the newest; the pilot app's anchors carry none, as the pilot's head is the view's turn.
"""

import numpy as np

from .scene import FloatArray

LONGEST_INTERVAL = 0.6  # Slower updates are a pause, not motion: show them at once.
LARGEST_STEP = 1.0  # Metres; a bigger move is a relocalization, not motion.


class AnchorGlide:
    def __init__(self):
        self.latest: FloatArray | None = None
        self.arrived = 0.0
        self.interval = 0.0
        self.start = self.target = np.zeros(3, dtype=np.float32)
        self.began = 0.0

    def position(self, now: float) -> FloatArray:
        if self.interval <= 0:
            return self.target
        progress = min(1.0, max(0.0, (now - self.began) / self.interval))
        return self.start + progress * (self.target - self.start)

    def __call__(self, anchor: FloatArray, now: float) -> FloatArray:
        """The anchor to draw at display time now; anchor is the newest from the link."""
        if anchor is not self.latest:
            target = np.asarray(anchor[:3, 3], dtype=np.float32)
            drawn = self.position(now)
            interval = now - self.arrived
            if (
                self.latest is None
                or interval > LONGEST_INTERVAL
                or np.linalg.norm(target - drawn) > LARGEST_STEP
            ):
                self.interval = 0.0
            else:
                # Smooth arrival jitter; a late update simply starts from where we are.
                self.interval = (
                    interval if self.interval <= 0 else 0.7 * self.interval + 0.3 * interval
                )
            self.start, self.target, self.began = drawn, target, now
            self.latest, self.arrived = anchor, now
        glided = np.array(anchor, dtype=np.float32)
        glided[:3, 3] = self.position(now)
        return glided


# Fusion's corrections reach the pilot no faster than this while their view moves, and not
# at all while it is still: below what people notice of a world turning or sliding under
# them (Steinicke et al., redirected-walking detection thresholds, IEEE TVCG 2010).
CORRECTION_TURN = np.radians(2.0)  # rad/s
CORRECTION_SLIDE = 0.05  # m/s
# The view counts as moving when the head turns or the eye travels this fast.
MOVING_TURN = np.radians(5.0)  # rad/s
MOVING_SLIDE = 0.02  # m/s
# A correction this large is a relocalization, not drift: shown at once.
LARGEST_TURN = np.radians(30.0)
LARGEST_SLIDE = 1.0


def _angle(rotation):
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))


def _toward_identity(rotation, angle):
    """Rotation turned toward identity by at most angle radians about its own axis."""
    full = _angle(rotation)
    if full <= angle or full < 1e-9:
        return np.eye(3)
    axis = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    ) / (2 * np.sin(full))
    keep = full - angle
    k = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(keep) * k + (1 - np.cos(keep)) * (k @ k)


class CorrectionEase:
    """Ease fusion's corrections into the pilot's view.

    Robot-first tracking places the eye at correction @ (the robot's camera pose), and new
    splats with the same correction. A new correction turns the newest scene about the eye
    (yaw drift) and moves the eye against the older scene (translation drift). Both are
    absorbed at once, so nothing jumps, and then played out at a rate a pilot does not
    notice, only while their view moves anyway: the world turns back about the eye, the eye
    slides to where fusion puts it.
    """

    def __init__(self):
        self.correction = None
        self.turn = np.eye(3)  # displayed world from fused world, about the eye
        self.slide = np.zeros(3)  # eye drawn from where fusion puts it, metres
        self.previous = None  # (time, head rotation, eye drawn)

    def __call__(self, correction, anchor, head, now):
        """The anchor to draw; anchor's position is the eye in the robot's odometry frame."""
        correction = np.asarray(correction, dtype=np.float64)
        robot = np.asarray(anchor[:3, 3], dtype=np.float64)
        eye = correction[:3, :3] @ robot + correction[:3, 3]
        if self.correction is not None and not np.array_equal(correction, self.correction):
            change = correction[:3, :3] @ self.correction[:3, :3].T
            self.turn = self.turn @ change.T
            before = self.correction[:3, :3] @ robot + self.correction[:3, 3]
            self.slide = self.slide + before - eye
        self.correction = correction
        head_rotation = None if head is None else np.asarray(head, np.float64)[:3, :3]
        drawn = eye + self.slide
        if _angle(self.turn) > LARGEST_TURN or np.linalg.norm(self.slide) > LARGEST_SLIDE:
            self.turn, self.slide = np.eye(3), np.zeros(3)
        elif self.previous is not None and now > self.previous[0]:
            dt = now - self.previous[0]
            turned = 0.0
            if head_rotation is not None and self.previous[1] is not None:
                turned = _angle(self.previous[1].T @ head_rotation) / dt
            travelled = np.linalg.norm(drawn - self.previous[2]) / dt
            if turned > MOVING_TURN or travelled > MOVING_SLIDE:
                self.turn = _toward_identity(self.turn, CORRECTION_TURN * dt)
                distance = np.linalg.norm(self.slide)
                if distance > 1e-9:
                    self.slide *= max(0.0, 1 - CORRECTION_SLIDE * dt / distance)
        drawn = eye + self.slide
        self.previous = (now, head_rotation, drawn)
        # The renderer draws world_from_eye = anchor @ head: a turn of the displayed world
        # about the eye is the inverse turn of the pilot's view.
        result = np.eye(4, dtype=np.float32)
        result[:3, :3] = self.turn.T @ np.asarray(anchor[:3, :3], np.float64)
        result[:3, 3] = drawn
        return result
